# A deliberately dense reference for regression-testing optimized scalar paths.
# It uses the original matrix definition, independent of the sparse kernels.
using Random

function dense_cell_polynomial(recon, u)
    n = length(u)
    middle = (n + 1) ÷ 2
    v = map(x -> x - u[middle], u)
    fits = recon.coefficients.fits
    optimal = RL._fit(fits.optimal, v)
    offsets = n == 3 ? (1:2, 2:3) : (1:3, 2:4, 3:5)
    lower = map(Base.tail(Tuple(fits)), offsets) do fit, indices
        a = RL._fit(fit, v[indices])
        ntuple(k -> k <= length(a) ? a[k] : zero(a[1]), n)
    end
    d = n == 3 ? (1 // 2, 1 // 4, 1 // 4) : (1 // 2, 1 // 6, 1 // 6, 1 // 6)
    p0 = ntuple(n) do j
        (optimal[j] - sum(d[k + 1] * lower[k][j] for k in eachindex(lower))) / d[1]
    end
    candidates = (p0, lower...)
    B = recon.coefficients.smoothness
    β = map(eachindex(candidates)) do k
        a = k == 1 && recon.weights isa ZWeights ? optimal : candidates[k]
        max(zero(a[1]), RL._tuple_dot(a, RL._fit(B, a)))
    end
    if recon.weights isa ZWeights
        τ = abs(RL._tuple_dot(optimal, RL._fit(recon.coefficients.tau, optimal)))
        power = recon.weights isa ZWeights{1} ? 1 : 2
        α = [d[k] * (1 + (τ / (β[k] + recon.epsilon))^power) for k in eachindex(d)]
    else
        α = [d[k] / (β[k] + recon.epsilon)^2 for k in eachindex(d)]
    end
    ω = α ./ sum(α)
    a = ntuple(j -> sum(ω[k] * candidates[k][j] for k in eachindex(candidates)), n)
    a0 = u[middle]
    if recon.input isa CellAverages
        a0 -= sum(a[k] / (k * 2^(k - 1)) for k in 3:2:n)
    end
    return (a0, Base.tail(a)...)
end

@testset "optimized CWENO agrees with dense reconstruction" begin
    rng = MersenneTwister(9013)
    for method in (CWENO3, CWENO5), input in (PointValues(), CellAverages()),
        weights in CWENO_WEIGHT_STRATEGIES, T in (Float32, Float64)
        recon = method(; input, weights, epsilon=T(1e-6))
        n = left_stencil_size(recon)
        for sample in 1:20
            # Include random smooth/rough values and small variations on an offset.
            u = sample <= 10 ? ntuple(_ -> T(randn(rng)), n) :
                ntuple(_ -> T(2) + T(1e-4) * T(randn(rng)), n)
            expected = dense_cell_polynomial(recon, u)
            actual = cell_polynomial(recon, u).coefficients
            @test collect(actual) ≈ collect(expected) rtol=128eps(T) atol=128eps(T)
        end
    end
end
