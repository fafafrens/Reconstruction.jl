using Printf
using Reconstruction

const HAS_PLOTS = try
    @eval import Plots
    true
catch err
    @warn "Plots.jl is not available; plot output will be skipped" exception=(err, catch_backtrace())
    false
end

const RECONSTRUCTIONS = (
    Godunov(),
    MinmodLimiter(),
    GeneralizedMinmodLimiter(theta=1.5),
    VanLeerLimiter(),
    VanAlbadaLimiter(),
    MonotonizedCentralLimiter(),
    SuperbeeLimiter(),
    WENO3(),
    WENOZ(),
    MP5(),
)

const PLOT_RECONSTRUCTIONS = (
    Godunov(),
    MinmodLimiter(),
    VanLeerLimiter(),
    SuperbeeLimiter(),
    WENO3(),
    WENOZ(),
    MP5(),
)

const DEMO_CASES = (
    (
        name="sine",
        title="Smooth sine",
        f=x -> sin(2π * x),
    ),
    (
        name="step",
        title="Step discontinuity",
        f=x -> x < 0.5 ? 1.0 : 0.0,
    ),
    (
        name="hat",
        title="Triangular hat",
        f=x -> max(0.0, 1.0 - abs(x - 0.5) / 0.2),
    ),
    (
        name="mixed",
        title="Smooth wave plus discontinuity",
        f=x -> sin(2π * x) + (0.35 <= x <= 0.65 ? 0.75 : 0.0),
    ),
)

reconstruction_label(recon) = string(nameof(typeof(recon)))
reconstruction_label(::MP5) = "MP5"

function reconstruct_periodic_faces(recon, u::AbstractVector)
    n = length(u)
    h = halo_width(recon)
    s = stencil_size(recon)
    left_states = Vector{float(eltype(u))}(undef, n)
    right_states = similar(left_states)

    @inbounds for i in 1:n
        stencil = ntuple(j -> u[mod1(i - h + j, n)], Val(s))
        left_states[i], right_states[i] = face(recon, stencil...)
    end

    return left_states, right_states
end

function ns_per_face(recon, u; repeats=500)
    left, right = reconstruct_periodic_faces(recon, u)
    start = time_ns()
    for _ in 1:repeats
        left, right = reconstruct_periodic_faces(recon, u)
    end
    return (time_ns() - start) / (repeats * length(u)), left, right
end

function comparison_table(case; n=512, repeats=500)
    xcell = ((0:n - 1) .+ 0.5) ./ n
    xface = (1:n) ./ n
    u = case.f.(xcell)

    println()
    println(case.title)
    @printf("%-28s %14s %14s %14s %14s\n", "reconstruction", "Linf left", "L1 left", "jump L1", "ns/face")
    @printf("%s\n", repeat("-", 90))

    for recon in RECONSTRUCTIONS
        timing, left, right = ns_per_face(recon, u; repeats)
        exact = case.f.(xface)
        err = abs.(left .- exact)
        jump = abs.(right .- left)
        @printf(
            "%-28s %14.6e %14.6e %14.6e %14.2f\n",
            reconstruction_label(recon),
            maximum(err),
            sum(err) / length(err),
            sum(jump) / length(jump),
            timing,
        )
    end
end

function maybe_make_plots(; n=256, outdir=joinpath(@__DIR__, "reconstruction_plots"))
    HAS_PLOTS || return String[]

    mkpath(outdir)
    xcell = ((0:n - 1) .+ 0.5) ./ n
    xface = (1:n) ./ n
    written = String[]

    for case in DEMO_CASES
        u = case.f.(xcell)
        p = Plots.plot(
            xcell,
            u;
            label="cell values",
            linewidth=3,
            xlabel="x",
            ylabel="left state",
            title="Left face states: $(case.title)",
            legend=:outerright,
        )

        for recon in PLOT_RECONSTRUCTIONS
            left, _ = reconstruct_periodic_faces(recon, u)
            Plots.plot!(p, xface, left; label=reconstruction_label(recon), linewidth=1.5)
        end

        path = joinpath(outdir, "$(case.name)_left_states.pdf")
        Plots.savefig(p, path)
        push!(written, path)
    end

    return written
end

function main()
    for case in DEMO_CASES
        comparison_table(case)
    end

    written = maybe_make_plots()
    if !isempty(written)
        @info "Wrote reconstruction plots" written
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
