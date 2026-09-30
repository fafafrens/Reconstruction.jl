# Run: julia --startup-file=no --project=. benchmarks_updates.jl [source-directory]
include("benchmarks/common.jl")

# Scalar Burgers Rusanov flux: both reconstructed states affect the work.
@inline update_flux(l, r) = (l * l + r * r) / 4 - max(abs(l), abs(r)) * (r - l) / 2
@inline update_face(recon, u, i) = face(recon, ntuple(k -> u[i + k - 1], Val(stencil_size(recon)))...)
@inline update_polynomial(recon, u, i) = cell_polynomial(recon,
    ntuple(k -> u[i + k - 1], Val(left_stencil_size(recon)))...)

@noinline function buffered_update!(out, flux, recon, u, dt_dx)
    N = length(out)
    radius = left_stencil_size(recon) ÷ 2
    @inbounds for i in 1:N
        flux[i] = update_flux(update_face(recon, u, i)...)
    end
    @inbounds for i in 1:N
        previous = i == 1 ? N : i - 1
        out[i] = u[i + radius] - dt_dx * (flux[i] - flux[previous])
    end
    return nothing
end

# The earlier fast cell sweep, followed by fluxes and the conservative update.
# Stores boundary values only, not the full polynomial coefficients.
@noinline function polynomial_buffered_update!(out, flux, l, r, recon, u, dt_dx)
    N = length(out)
    radius = left_stencil_size(recon) ÷ 2
    @inbounds for i in 2:N
        p = update_polynomial(recon, u, i)
        l[i] = value(p, 1 // 2)
        r[i - 1] = value(p, -1 // 2)
    end
    @inbounds begin
        p = update_polynomial(recon, u, 1)
        l[1] = value(p, 1 // 2)
        r[N] = value(p, -1 // 2)
    end
    @inbounds for i in 1:N
        flux[i] = update_flux(l[i], r[i])
    end
    @inbounds for i in 1:N
        previous = i == 1 ? N : i - 1
        out[i] = u[i + radius] - dt_dx * (flux[i] - flux[previous])
    end
    return nothing
end

# Visit right faces in order, retaining the preceding flux; write each cell once.
@noinline function rolling_update!(out, recon, u, dt_dx)
    N = length(out)
    radius = left_stencil_size(recon) ÷ 2
    boundary = update_flux(update_face(recon, u, N)...)
    previous = boundary
    @inbounds for i in 1:(N - 1)
        current = update_flux(update_face(recon, u, i)...)
        out[i] = u[i + radius] - dt_dx * (current - previous)
        previous = current
    end
    @inbounds out[N] = u[N + radius] - dt_dx * (boundary - previous)
    return nothing
end

# Each unique periodic face adds opposite contributions to its two neighbors.
# This accumulation loop is serial: adjacent faces share output cells.
@noinline function accumulating_update!(out, recon, u, dt_dx)
    N = length(out)
    radius = left_stencil_size(recon) ÷ 2
    copyto!(out, 1, u, radius + 1, N)
    @inbounds for i in 1:N
        next = i == N ? 1 : i + 1
        contribution = dt_dx * update_flux(update_face(recon, u, i)...)
        out[i] -= contribution
        out[next] += contribution
    end
    return nothing
end

@noinline function polynomial_update!(out, centers, gradients, recon, u, dt_dx, dx, ::Val{C}) where {C}
    N = length(out)
    radius = left_stencil_size(recon) ÷ 2
    first = update_polynomial(recon, u, 1)
    last = N == 1 ? first : update_polynomial(recon, u, N)
    boundary = update_flux(face(last, first)...)
    previous = boundary
    p = first
    @inbounds for i in 1:(N - 1)
        q = i == N - 1 ? last : update_polynomial(recon, u, i + 1)
        current = update_flux(face(p, q)...)
        out[i] = u[i + radius] - dt_dx * (current - previous)
        if C
            centers[i], gradients[i] = center(p; dx)
        end
        previous, p = current, q
    end
    @inbounds begin
        out[N] = u[N + radius] - dt_dx * (boundary - previous)
        if C
            centers[N], gradients[N] = center(p; dx)
        end
    end
    return nothing
end

@noinline function polynomial_accumulating_update!(out, recon, u, dt_dx)
    N = length(out)
    radius = left_stencil_size(recon) ÷ 2
    copyto!(out, 1, u, radius + 1, N)
    first = update_polynomial(recon, u, 1)
    p = first
    @inbounds for i in 1:N
        next = i == N ? 1 : i + 1
        q = i == N ? first : update_polynomial(recon, u, next)
        contribution = dt_dx * update_flux(face(p, q)...)
        out[i] -= contribution
        out[next] += contribution
        p = q
    end
    return nothing
end

function update_cases()
    return (
        ("Godunov", Godunov(), "averages"),
        ("Minmod", MinmodLimiter(), "averages"),
        ("GeneralizedMinmod", GeneralizedMinmodLimiter(), "averages"),
        ("VanLeer", VanLeerLimiter(), "averages"),
        ("VanAlbada", VanAlbadaLimiter(), "averages"),
        ("MC", MonotonizedCentralLimiter(), "averages"),
        ("Superbee", SuperbeeLimiter(), "averages"),
        ("MP5", MP5(), "averages"),
        ("PPM", PPM(), "averages"),
        ("WENO3", WENO3(), "points"),
        ("WENO-Z", WENOZ(), "points"),
        ("CWENO3", CWENO3(), "averages"),
        ("CWENO5", CWENO5(), "averages"),
        ("CWENO-Z3", CWENO3(; weights=ZWeights()), "averages"),
        ("CWENO-Z5", CWENO5(; weights=ZWeights()), "averages"),
        ("CWENO-Z3", CWENO3(; weights=ZWeights(), input=PointValues()), "points"),
        ("CWENO-Z5", CWENO5(; weights=ZWeights(), input=PointValues()), "points"),
    )
end

function update_workloads(recon, input, profile, N)
    u = periodic_samples(recon, input, profile, N)
    out, flux, centers, gradients, l, r = ntuple(_ -> zeros(N), 6)
    dx = 1 / N
    radius = left_stencil_size(recon) ÷ 2
    # A fixed old state is used for every repetition, with a conservative CFL.
    dt_dx = 0.1 / max(1.0, maximum(abs, u))
    raw = u[(radius + 1):(radius + N)]
    # Independent reference: periodic indexing rather than padded stencil indexing.
    reference_flux = [update_flux(face(recon,
        ntuple(k -> raw[mod1(i - radius + k - 1, N)], stencil_size(recon))...)...)
        for i in 1:N]
    expected = [raw[i] - dt_dx * (reference_flux[i] - reference_flux[mod1(i - 1, N)]) for i in 1:N]
    runs = (
        () -> buffered_update!(out, flux, recon, u, dt_dx),
        () -> rolling_update!(out, recon, u, dt_dx),
        () -> accumulating_update!(out, recon, u, dt_dx),
    )
    polynomial = recon isa Union{Godunov,AbstractSlopeLimiter,CWENO3,CWENO5,PPM}
    if polynomial
        runs = (runs...,
            () -> polynomial_update!(out, centers, gradients, recon, u, dt_dx, dx, Val(false)),
            () -> polynomial_accumulating_update!(out, recon, u, dt_dx),
            () -> polynomial_buffered_update!(out, flux, l, r, recon, u, dt_dx),
            () -> polynomial_update!(out, centers, gradients, recon, u, dt_dx, dx, Val(true)))
    end
    for run in runs
        run()
        @assert all(isfinite, out)
        @assert isapprox(out, expected; atol=1e-12, rtol=1e-12)
        @assert abs(sum(out) - sum(raw)) <= 256eps(Float64) * max(1.0, sum(abs, raw))
    end
    if polynomial
        for i in 1:N
            expected_center = center(recon,
                ntuple(k -> raw[mod1(i - radius + k - 1, N)], left_stencil_size(recon)); dx)
            @assert isapprox(centers[i], expected_center.value; atol=1e-12, rtol=1e-12)
            @assert isapprox(gradients[i], expected_center.derivative; atol=1e-12, rtol=1e-12)
        end
    end
    @assert u == periodic_samples(recon, input, profile, N)
    return runs
end

function benchmark_updates(; N=4096, repetitions=100, samples=15)
    cases = update_cases()
    # Exercise wraparound separately, including domains smaller than the stencil.
    for (_, recon, input) in cases, size in (1, 2, 17)
        update_workloads(recon, input, "mixed", size)
    end
    println("Source: ", BENCHMARK_SOURCE)
    println("Julia ", VERSION, "; ", Sys.CPU_NAME, "; threads=", Threads.nthreads())
    println(N, " periodic Float64 cells; microseconds per complete scalar Burgers update")
    println("Includes polynomial construction, fluxes and output writes; excludes halo setup, reconstruction-object setup and compilation.")
    println("Point-input rows compare work only, not a cell-average finite-volume discretization.")
    @printf("%-7s %-19s %-8s %10s %10s %10s %10s %10s %10s %10s\n",
        "Profile", "Method", "Input", "Buffered", "Face roll", "Face add", "Poly roll", "Poly add", "Poly batch", "+ center")
    for profile in ("smooth", "mixed", "rough"), (label, recon, input) in cases
        runs = update_workloads(recon, input, profile, N)
        results = map(f -> measure_sweep(f; repetitions, samples), runs)
        @assert all(r -> r.bytes == 0, results)
        @printf("%-7s %-19s %-8s", profile, label, input)
        for result in results
            @printf(" %10.2f", result.microseconds)
        end
        for _ in (length(results) + 1):7
            @printf(" %10s", "—")
        end
        println()
        flush(stdout)
    end
    println("All measured updates allocate zero bytes. + center includes Poly roll and old-state center values/gradients.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_updates()
end
