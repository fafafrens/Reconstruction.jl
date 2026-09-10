# Included by runtests_cell_polynomials.jl.

@testset "CWENO-Z global indicators" begin
    for input in (PointValues(), CellAverages()), (method, n) in ((CWENO3, 3), (CWENO5, 5))
        recon = method(; input, weights=ZWeights())
        fits = RL._exact_fits(input, Val(n))
        B = RL._exact_smoothness(Val(n))
        Q = RL._exact_tau(input, Val(n), fits, B)
        expected_order = n == 3 ? 4 : 6
        # Coefficient a_k of a smooth local polynomial is O(dx^k). Verify exact
        # cancellation of every quadratic term below the claimed order of tau.
        @test all(Q[i][j] == 0 for i in 1:n, j in 1:n if i + j - 2 < expected_order)
        @test any(Q[i][j] != 0 for i in 1:n, j in 1:n if i + j - 2 == expected_order)

        # Compare the precombined form with the defining indicator combination
        # using exact sample values and independent stencil fits.
        a = ntuple(k -> (-1)^k * k // 7, n)
        stencils = n == 3 ? (-1:0, 0:1) : (-2:0, -1:1, 0:2)
        factors = n == 3 ? (1, 1) : (input isa CellAverages ? (1, 4, 1) : (1, 6, 1))
        βopt = RL._smoothness(B, a)
        reference = -sum(factors) * βopt
        for (st, factor) in zip(stencils, factors)
            samples = ntuple(length(st)) do i
                x = st[i] // 1
                sum(1:n) do k
                    moment = input isa PointValues ? x^(k - 1) :
                        ((x + 1 // 2)^k - (x - 1 // 2)^k) / k
                    a[k] * moment
                end
            end
            low = RL._fit(RL._exact_fit(input, st), samples)
            padded = ntuple(k -> k <= length(low) ? low[k] : 0 // 1, n)
            reference += factor * RL._smoothness(B, padded)
        end
        @test RL._tuple_dot(a, RL._fit(Q, a)) == reference
        @test RL._cweno_tau(recon, Float64.(a)) ≈ abs(reference)

        # The central weight must use Popt, not the additional candidate P0.
        optimal = Float64.(a)
        m = n == 3 ? 3 : 4
        candidates = ntuple(k -> ntuple(j -> 0.1 * j * k, n), m)
        modified = Base.setindex(candidates, ntuple(_ -> 100.0, n), 1)
        d = n == 3 ? (0.5, 0.25, 0.25) : (0.5, 1 / 6, 1 / 6, 1 / 6)
        @test RL._cweno_weights(recon, optimal, candidates, d) ==
            RL._cweno_weights(recon, optimal, modified, d)
    end
end

@testset "CWENO-Z normalization and defaults" begin
    for power in (1, 2), d in ((0.5, 0.25, 0.25), (0.5, 1 / 6, 1 / 6, 1 / 6))
        weights = ZWeights(; power)
        β = ntuple(k -> 0.1k^2, length(d))
        τ, epsilon = 0.3, 1e-6
        α = map((dk, b) -> dk * (1 + (τ / (b + epsilon))^power), d, β)
        expected = map(a -> a / sum(α), α)
        actual = RL._z_normalized_weights(weights, β, τ, epsilon, d)
        @test collect(actual) ≈ collect(expected)
        @test collect(RL._z_normalized_weights(weights, β, 0.0, epsilon, d)) ≈ collect(d)

        # The direct formula would overflow, but normalized weights are finite.
        extreme = RL._z_normalized_weights(weights,
            ntuple(k -> Float64(k - 1), length(d)), 1e100, 1e-300, d)
        @test all(isfinite, extreme)
        @test sum(extreme) ≈ 1
        @test extreme[1] ≈ 1
    end
    for power in (-1, 0, 3)
        @test_throws ArgumentError ZWeights(; power)
        @test_throws ArgumentError ZWeights{power}()
    end
    for method in (CWENO3, CWENO5), input in (PointValues(), CellAverages())
        default = method(; input)
        @test default.weights isa JSWeights
        @test default.coefficients.tau === nothing
        n = left_stencil_size(default)
        u = ntuple(k -> sin(k), n)
        @test cell_polynomial(default, u).coefficients ==
            cell_polynomial(method(; input, weights=JSWeights()), u).coefficients
        for power in (1, 2)
            z = method(; input, weights=ZWeights(; power), epsilon=1e-300)
            p = cell_polynomial(z, ntuple(k -> k <= (n + 1) ÷ 2 ? 1.0 : 0.0, n))
            @test all(isfinite, p.coefficients)
        end
    end
end

@testset "CWENO-Z higher critical points" begin
    for weights in (ZWeights(), ZWeights(; power=1)), input in (PointValues(), CellAverages()),
        critical in (2, 3)
        errors = [cweno_errors(input, h; weights, critical) for h in (0.1, 0.05, 0.025)]
        for j in 1:2, level in 1:2
            order = log2(errors[level][j] / errors[level + 1][j])
            @test order > (j == 1 ? 4.7 : 3.7)
        end
    end
end
