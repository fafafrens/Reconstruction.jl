# Run: julia --project=. benchmarks_cweno.jl [earlier-source-directory]
# The optional source directory allows comparison with an earlier checkout.
include("benchmarks/common.jl")

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
    @printf("%-22s %12s %12s %12s %12s\n", "Profile / method", "WENO-Z", "CWENO-Z", "Reuse AoS", "Reuse SoA")
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
            wz = WENOZ()
            weno = measure_sweep(() -> direct_faces!(l, r, wz, u))
            direct = measure_sweep(() -> direct_faces!(l, r, recon, u))
            reuse = measure_sweep(() -> reuse_faces!(l, r, polynomials, recon, u))
            soa = measure_sweep(() -> reuse_coefficients!(l, r, coefficients, recon, u))
            @assert weno.bytes == direct.bytes == reuse.bytes == soa.bytes == 0
            label = string(discontinuous ? "mixed" : "smooth", ", power=", power)
            @printf("%-22s %12.2f %12.2f %12.2f %12.2f\n", label,
                weno.microseconds, direct.microseconds, reuse.microseconds, soa.microseconds)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_cweno()
end
