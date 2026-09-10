# Included by runtests_reconstruction_limiters.jl.
const RL = Reconstruction
const CWENO_WEIGHT_STRATEGIES = (JSWeights(), ZWeights(), ZWeights(; power=1))

# Four-point Gauss quadrature on [-1/2, 1/2], exact through degree seven.
const GAUSS_NODES = (
    -sqrt((3 + 2sqrt(6 / 5)) / 7) / 2,
    -sqrt((3 - 2sqrt(6 / 5)) / 7) / 2,
    sqrt((3 - 2sqrt(6 / 5)) / 7) / 2,
    sqrt((3 + 2sqrt(6 / 5)) / 7) / 2,
)
const GAUSS_WEIGHTS = (
    (18 - sqrt(30)) / 72, (18 + sqrt(30)) / 72,
    (18 + sqrt(30)) / 72, (18 - sqrt(30)) / 72,
)
cell_integral(p) = sum(w * value(p, x) for (x, w) in zip(GAUSS_NODES, GAUSS_WEIGHTS))

@testset "shared polynomial interface" begin
    p = CellPolynomial((2, 3.0, 4))
    @test value(p, 0.25) == 3.0
    @test derivative(p, 0.25; dx=0.5) == 10.0
    @test center(p; dx=0.5) == (value=2.0, derivative=6.0)
    @test_throws ArgumentError CellPolynomial(())
    for dx in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError derivative(p, 0; dx)
        @test_throws ArgumentError center(Godunov(), 1.0; dx)
    end

    p0 = cell_polynomial(Godunov(), 2.0)
    @test p0.coefficients == (2.0,)
    @test derivative(p0, 0.3; dx=0.2) == 0.0
    @test face(p0, cell_polynomial(Godunov(), 3.0)) == (2.0, 3.0)

    for lim in filter(r -> r isa AbstractSlopeLimiter, RECONS)
        p = @inferred cell_polynomial(lim, 0.0, 1.0, 2.0)
        @test center(p; dx=0.25) == (value=1.0, derivative=4.0)
        @test cell_integral(p) ≈ 1.0
        @test value(p, 0.5) == left(lim, 0.0, 1.0, 2.0)
        @test value(p, -0.5) == right(lim, 0.0, 1.0, 2.0)
        @test center(lim, (0.0, 1.0, 2.0); dx=0.25) == center(p; dx=0.25)
        @test center(lim, [0.0, 1.0, 2.0]; dx=0.25) == center(p; dx=0.25)
        @test center(lim, 0.0, 1.0, 0.0; dx=0.25).derivative == 0.0
    end
    @test_throws DimensionMismatch cell_polynomial(MinmodLimiter(), [1.0, 2.0])
    @test_throws MethodError cell_polynomial(WENOZ(), 1.0, 2.0, 3.0, 4.0, 5.0)
end

@testset "CWENO coefficient tables" begin
    for method in (CWENO3, CWENO5), input in (PointValues(), CellAverages())
        recon = method(; input)
        @test isbitstype(typeof(recon))
        stencils = method === CWENO3 ? (-1:1, -1:0, 0:1) : (-2:2, -2:0, -1:1, 0:2)
        for (table, offsets) in zip(recon.coefficients.fits, stencils)
            n = length(offsets)
            # Each monomial is recovered from its point samples or integrals.
            for degree in 0:(n - 1)
                samples = ntuple(n) do k
                    x = offsets[k]
                    input isa PointValues ? Float64(x^degree) :
                        ((x + 0.5)^(degree + 1) - (x - 0.5)^(degree + 1)) / (degree + 1)
                end
                a = RL._fit(table, samples)
                @test collect(a) ≈ [j == degree + 1 ? 1.0 : 0.0 for j in 1:n] atol=2e-14
            end
        end

        # Independent quadrature of the squared polynomial derivatives.
        a = method === CWENO3 ? (1.3, -0.7, 0.4) : (1.3, -0.7, 0.4, -0.2, 0.1)
        degree = length(a) - 1
        β = sum(1:degree) do l
            sum(zip(GAUSS_NODES, GAUSS_WEIGHTS)) do (x, w)
                dp = sum(a[k + 1] * factorial(k) / factorial(k - l) * x^(k - l) for k in l:degree)
                w * dp^2
            end
        end
        @test RL._smoothness(recon.coefficients.smoothness, a) ≈ β
    end
    for method in (CWENO3, CWENO5), epsilon in (0.0, -1.0, NaN, Inf)
        @test_throws ArgumentError method(; epsilon)
    end
end

@testset "CWENO conservation, interpolation, and symmetry" begin
    for weights in CWENO_WEIGHT_STRATEGIES, input in (PointValues(), CellAverages())
        recon = CWENO5(; input, weights)
        for u in ((-0.7, 1.2, 0.4, 2.1, -1.3), (1.0, 1.0, 0.3, 0.0, 0.0))
            p = @inferred cell_polynomial(recon, u...)
            reflected = cell_polynomial(recon, reverse(u))
            if input isa CellAverages
                @test cell_integral(p) ≈ u[3] atol=1e-14
            else
                @test value(p, 0) == u[3]
            end
            for x in (-0.5, -0.2, 0.0, 0.3, 0.5)
                @test isfinite(value(p, x))
                @test isfinite(derivative(p, x; dx=0.1))
                @test value(p, x) ≈ value(reflected, -x) atol=1e-14
            end
            @test value(p, 0.5) ≈ left(recon, u...)
            @test value(p, -0.5) ≈ right(recon, u...)
            @test collect(cell_polynomial(recon, collect(u)).coefficients) ≈ collect(p.coefficients) rtol=16eps() atol=16eps()
        end

        # All candidate polynomials reproduce this quadratic.
        u = ntuple(k -> (k - 3)^2 + 0.7 * (k - 3) + 2 +
            (input isa CellAverages ? 1 / 12 : 0), 5)
        p = cell_polynomial(recon, u)
        @test collect(p.coefficients) ≈ [2.0, 0.7, 1.0, 0.0, 0.0] atol=1e-14
        @test center(recon, u; dx=0.2).derivative ≈ 3.5

        # A smooth constant substencil should dominate when the jump is outside
        # the reconstructed cell. No TVD assertion is made for a jump inside it.
        p = cell_polynomial(CWENO5(; input, weights, epsilon=1e-10), 1.0, 1.0, 1.0, 0.0, 0.0)
        @test maximum(abs(value(p, x) - 1) for x in GAUSS_NODES) < 1e-9
        constant = cell_polynomial(recon, ntuple(_ -> 1.2345, 5))
        @test center(constant; dx=0.1) == (value=1.2345, derivative=0.0)

        st = (0.7, -1.2, 0.4, 2.1, 0.8, -0.6)
        p = cell_polynomial(recon, st[1:5])
        q = cell_polynomial(recon, st[2:6])
        @test collect(face(p, q)) ≈ collect(face(recon, st...))
        @test collect(face(recon, st)) ≈ collect(face(recon, collect(st))) rtol=16eps() atol=16eps()
        arrays = map(x -> [x, 2x], st)
        l, r = zeros(2), zeros(2)
        face!(recon, l, r, arrays...)
        expected_l, expected_r = face(recon, arrays...)
        @test l ≈ expected_l rtol=16eps() atol=16eps()
        @test r ≈ expected_r rtol=16eps() atol=16eps()
        @test [l[1], r[1]] ≈ collect(face(recon, st...)) rtol=16eps() atol=16eps()
        @test_throws DimensionMismatch cell_polynomial(recon, ones(4))
        @test_throws DimensionMismatch cell_polynomial(recon, ones(6))
        @test_throws MethodError cell_polynomial(recon, 1.0, 2.0, 3.0)
    end
end

@testset "CWENO3 conservation, interpolation, and interfaces" begin
    for weights in CWENO_WEIGHT_STRATEGIES, input in (PointValues(), CellAverages())
        recon = CWENO3(; input, weights)
        for u in ((-0.7, 1.2, 0.4), (1.0, 0.3, 0.0))
            p = @inferred cell_polynomial(recon, u...)
            reflected = cell_polynomial(recon, reverse(u))
            @test p isa CellPolynomial{3}
            if input isa CellAverages
                @test cell_integral(p) ≈ u[2] atol=1e-14
            else
                @test value(p, 0) == u[2]
            end
            for x in (-0.5, -0.2, 0.0, 0.3, 0.5)
                @test isfinite(value(p, x))
                @test isfinite(derivative(p, x; dx=0.1))
                @test value(p, x) ≈ value(reflected, -x) atol=1e-14
            end
            @test value(p, 0.5) ≈ left(recon, u...)
            @test value(p, -0.5) ≈ right(recon, u...)
            @test collect(cell_polynomial(recon, collect(u)).coefficients) ≈ collect(p.coefficients) rtol=16eps() atol=16eps()
            @test collect(center(recon, u; dx=0.2)) ≈ collect(center(p; dx=0.2)) rtol=16eps() atol=16eps()
            @test collect(center(recon, collect(u); dx=0.2)) ≈ collect(center(p; dx=0.2)) rtol=16eps() atol=16eps()
        end
        # A jump outside the reconstructed cell leaves one smooth linear candidate.
        p = cell_polynomial(CWENO3(; input, weights, epsilon=1e-10), 1.0, 1.0, 0.0)
        @test maximum(abs(value(p, x) - 1) for x in GAUSS_NODES) < 1e-9
        constant = cell_polynomial(recon, 1.2345, 1.2345, 1.2345)
        @test center(constant; dx=0.1) == (value=1.2345, derivative=0.0)
        @test center(recon, 0.0, 1.0, 2.0; dx=0.25) == (value=1.0, derivative=4.0)

        st = (0.7, -1.2, 0.4, 2.1)
        p = cell_polynomial(recon, st[1:3])
        q = cell_polynomial(recon, st[2:4])
        @test collect(face(p, q)) ≈ collect(face(recon, st...))
        @test collect(face(recon, st)) ≈ collect(face(recon, collect(st))) rtol=16eps() atol=16eps()
        arrays = map(x -> [x, 2x], st)
        l, r = zeros(2), zeros(2)
        face!(recon, l, r, arrays...)
        expected_l, expected_r = face(recon, arrays...)
        @test l ≈ expected_l rtol=16eps() atol=16eps()
        @test r ≈ expected_r rtol=16eps() atol=16eps()
        @test [l[1], r[1]] ≈ collect(face(recon, st...)) rtol=16eps() atol=16eps()
        @test_throws DimensionMismatch cell_polynomial(recon, ones(2))
        @test_throws DimensionMismatch cell_polynomial(recon, ones(4))
        @test_throws MethodError cell_polynomial(recon, 1.0, 2.0)
        @test_throws MethodError face(recon, 1.0, 2.0, 3.0)
        @test_throws MethodError face!(recon, l, r, arrays..., arrays[1])
    end
end

function cweno_exp_sample(input, x, h, critical)
    factor = input isa CellAverages ? sinh(h / 2) / (h / 2) : 1.0
    result = exp(x) * factor
    for degree in 1:Int(critical)
        moment = input isa PointValues ? x^degree :
            ((x + h / 2)^(degree + 1) - (x - h / 2)^(degree + 1)) / (h * (degree + 1))
        result -= moment / factorial(degree)
    end
    return result
end

function cweno_errors(input, h; method=CWENO5, weights=JSWeights(), critical=false, epsilon=h^2)
    # Removing Taylor terms creates critical points of the specified order.
    recon = method(; input, epsilon, weights)
    n = left_stencil_size(recon)
    samples = ntuple(n) do k
        x = (k - (n + 1) ÷ 2) * h
        cweno_exp_sample(input, x, h, critical)
    end
    p = cell_polynomial(recon, samples)
    value_error = maximum((-0.5, 0.0, 0.2, 0.5)) do ξ
        abs(value(p, ξ) - cweno_exp_sample(PointValues(), h * ξ, h, critical))
    end
    derivative_error = abs(derivative(p, 0; dx=h) - (critical > 0 ? 0 : 1))
    return value_error, derivative_error
end

@testset "CWENO smooth convergence" begin
    for weights in CWENO_WEIGHT_STRATEGIES, method in (CWENO3, CWENO5),
        input in (PointValues(), CellAverages()), critical in (false, true)
        order_target = method === CWENO3 ? 3 : 5
        errors = [cweno_errors(input, h; method, weights, critical) for h in (0.1, 0.05, 0.025)]
        for j in 1:2, level in 1:2
            order = log2(errors[level][j] / errors[level + 1][j])
            @test order > (j == 1 ? order_target - 0.3 : order_target - 1.3)
        end
        # Exercise the default fixed epsilon as well as the mesh-scaled choice.
        coarse = cweno_errors(input, 0.05; method, weights, critical, epsilon=1e-6)
        fine = cweno_errors(input, 0.025; method, weights, critical, epsilon=1e-6)
        @test all(fine .< coarse)
    end

    # Check center values separately: point-input interpolation is exact there,
    # whereas average-input reconstruction must recover a different value.
    for weights in CWENO_WEIGHT_STRATEGIES, method in (CWENO3, CWENO5), critical in (false, true)
        order_target = method === CWENO3 ? 3 : 5
        center_errors = map((0.1, 0.05, 0.025)) do h
            n = order_target
            samples = ntuple(n) do k
                x = (k - (n + 1) ÷ 2) * h
                exp(x) * sinh(h / 2) / (h / 2) - (critical ? x : 0)
            end
            result = center(method(; weights, epsilon=h^2), samples; dx=h)
            abs(result.value - 1)
        end
        @test all(log2(center_errors[j] / center_errors[j + 1]) > order_target - 0.3 for j in 1:2)
    end
end

polynomial_allocations(recon, u) = @allocated cell_polynomial(recon, u...)

@testset "CWENO numeric types and allocations" begin
    for weights in CWENO_WEIGHT_STRATEGIES, method in (CWENO3, CWENO5),
        T in (Float32, Float64), input in (PointValues(), CellAverages())
        recon = @inferred method(; input, weights, epsilon=T(1e-6))
        n = left_stencil_size(recon)
        u = (T(0.2), T(0.4), T(0.5), T(0.8), T(0.9))[1:n]
        p = @inferred cell_polynomial(recon, u...)
        c = @inferred center(recon, u; dx=T(0.1))
        @test p isa CellPolynomial{n,T}
        @test c.value isa T
        @test c.derivative isa T
        @test value(p, T(0.5)) isa T
        @test left(recon, u...) isa T
        polynomial_allocations(recon, u) # Warm the measurement wrapper.
        @test polynomial_allocations(recon, u) == 0
    end
end

include(joinpath(@__DIR__, "runtests_cweno_z.jl"))
include(joinpath(@__DIR__, "runtests_cweno_reference.jl"))
