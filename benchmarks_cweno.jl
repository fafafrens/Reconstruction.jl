# Run: julia --project=. benchmarks_cweno.jl [earlier-source-directory]
# The optional source directory allows comparison with an earlier checkout.
include("benchmarks/common.jl")

# Evaluate both boundaries immediately; retain only the final face states.
@noinline function reuse_boundaries!(l, r, recon, u)
    n = Val(left_stencil_size(recon))
    @inbounds for i in 2:length(l)
        p = cell_polynomial(recon, ntuple(k -> u[i + k - 1], n)...)
        l[i] = value(p, 1 // 2)
        r[i - 1] = value(p, -1 // 2)
    end
    @inbounds begin
        p = cell_polynomial(recon, ntuple(k -> u[k], n)...)
        l[1] = value(p, 1 // 2)
        r[end] = value(p, -1 // 2)
    end
    return nothing
end

@noinline function reuse_faces!(l, r, polynomials, recon, u)
    @inbounds for i in eachindex(l)
        stencil = ntuple(k -> u[i + k - 1], Val(5))
        polynomials[i] = cell_polynomial(recon, stencil...)
    end
    @inbounds for i in eachindex(l)
        j = i == length(l) ? firstindex(l) : i + 1
        l[i], r[i] = face(polynomials[i], polynomials[j])
    end
    return nothing
end

# Separate coefficient arrays permit vectorization across neighboring cells.
# CellPolynomial remains the scalar interface; no new representation is needed
# inside the numerical reconstruction itself.
@noinline function reuse_coefficients!(l, r, coefficients, recon, u)
    @inbounds for i in eachindex(l)
        p = cell_polynomial(recon, ntuple(k -> u[i + k - 1], Val(5)))
        ntuple(k -> coefficients[k][i] = p.coefficients[k], Val(5))
    end
    @inbounds for i in eachindex(l)
        j = i == length(l) ? firstindex(l) : i + 1
        p = CellPolynomial(ntuple(k -> coefficients[k][i], Val(5)))
        q = CellPolynomial(ntuple(k -> coefficients[k][j], Val(5)))
        l[i], r[i] = face(p, q)
    end
    return nothing
end

function benchmark_cweno(; N=4096)
    @printf("%d periodic Float64 cells; median microseconds per complete face sweep\n", N)
    @printf("%-22s %12s %12s %12s %12s %12s\n", "Profile / method", "WENO-Z", "CWENO-Z", "Reuse AoS", "Reuse SoA", "Reuse local")
    for discontinuous in (false, true)
        u = periodic_samples(WENOZ(), "points", discontinuous ? "mixed" : "smooth", N)
        for power in (1, 2)
            recon = CWENO5(; input=PointValues(), weights=ZWeights(; power))
            l, r = zeros(N), zeros(N)
            polynomials = Vector{typeof(cell_polynomial(recon, u[1:5]))}(undef, N)
            coefficients = ntuple(_ -> zeros(N), Val(5))
            direct_faces!(l, r, recon, u)
            expected_l, expected_r = copy(l), copy(r)
            reuse_faces!(l, r, polynomials, recon, u)
            @assert isapprox(l, expected_l) && isapprox(r, expected_r)
            reuse_coefficients!(l, r, coefficients, recon, u)
            @assert isapprox(l, expected_l) && isapprox(r, expected_r)
            reuse_boundaries!(l, r, recon, u)
            @assert isapprox(l, expected_l) && isapprox(r, expected_r)
            wz = WENOZ()
            weno = measure_sweep(() -> direct_faces!(l, r, wz, u))
            direct = measure_sweep(() -> direct_faces!(l, r, recon, u))
            reuse = measure_sweep(() -> reuse_faces!(l, r, polynomials, recon, u))
            soa = measure_sweep(() -> reuse_coefficients!(l, r, coefficients, recon, u))
            local_reuse = measure_sweep(() -> reuse_boundaries!(l, r, recon, u))
            @assert local_reuse.bytes == 0
            @assert weno.bytes == direct.bytes == reuse.bytes == soa.bytes == 0
            label = string(discontinuous ? "mixed" : "smooth", ", power=", power)
            @printf("%-22s %12.2f %12.2f %12.2f %12.2f %12.2f\n", label,
                weno.microseconds, direct.microseconds, reuse.microseconds, soa.microseconds, local_reuse.microseconds)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_cweno()
end
