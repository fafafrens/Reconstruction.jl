# Independent boundary-value form of classic PPM, with the q6 limiter.
function ppm_reference(stencil)
    s = map(2:4) do i
        dm, dp = stencil[i] - stencil[i - 1], stencil[i + 1] - stencil[i]
        dc = (stencil[i + 1] - stencil[i - 1]) / 2
        dm * dp <= 0 ? zero(dc) : sign(dc) * min(abs(dc), 2abs(dm), 2abs(dp))
    end
    u = stencil[3]
    l = (stencil[2] + u) / 2 - (s[2] - s[1]) / 6
    r = (u + stencil[4]) / 2 - (s[3] - s[2]) / 6
    if (r - u) * (u - l) <= 0
        l = r = u
    else
        delta = r - l
        q6 = 6u - 3(l + r)
        if delta * q6 > delta^2
            l = 3u - 2r
        elseif delta * q6 < -delta^2
            r = 3u - 2l
        end
    end
    return x -> l + (x + 1 / 2) * (r - l + (6u - 3(l + r)) * (1 / 2 - x))
end

function ppm_allocation_check(stencil, dx)
    recon = PPM()
    cell_polynomial(recon, stencil...)
    center(recon, stencil...; dx)
    left(recon, stencil...)
    return (@allocated(cell_polynomial(recon, stencil...)),
        @allocated(center(recon, stencil...; dx)), @allocated(left(recon, stencil...)))
end

@testset "PPM interfaces and exact reconstruction" begin
    recon = PPM()
    @test isbitstype(typeof(recon))
    @test (halo_width(recon), stencil_size(recon), left_stencil_size(recon)) == (3, 6, 5)
    # Cell averages of a monotone quadratic; limiting should be inactive.
    averages = ntuple(k -> 10 + 3(k - 3) + (1 // 10) * ((k - 3)^2 + 1 // 12), 6)
    p = cell_polynomial(recon, averages[1:5])
    @test p.coefficients == (10, 3, 1 // 10)
    @test value(p, 1 // 2) == left(recon, averages[1:5]...)
    @test value(p, -1 // 2) == right(recon, averages[1:5]...)
    @test face(recon, averages) == face(p, cell_polynomial(recon, averages[2:6]))
    @test face(recon, collect(averages)) == face(recon, averages...)
    @test cell_polynomial(recon, collect(averages[1:5])) == p
    @test center(recon, averages[1:5]; dx=1) == (value=10, derivative=3)
    @test center(recon, collect(averages[1:5]); dx=1) == center(p; dx=1)
    l, r = zeros(2), zeros(2)
    arrays = map(v -> fill(Float64(v), 2), averages)
    @test face!(recon, l, r, arrays...) === (l, r)
    @test all(isapprox.(l, Float64(left(recon, averages[1:5]...))))
    @test all(isapprox.(r, Float64(right(recon, averages[2:6]...))))
    @test_throws DimensionMismatch cell_polynomial(recon, [1.0, 2.0, 3.0])
    @test_throws BoundsError face(recon, ones(5))
    @test_throws MethodError cell_polynomial(recon, 1.0, 2.0, 3.0)
    for dx in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError center(recon, averages[1:5]; dx)
    end
end

@testset "PPM limiting, conservation, symmetry and reference" begin
    rng = MersenneTwister(4831)
    for T in (Float32, Float64)
        tolerance = 64eps(T)
        stencils = [ntuple(_ -> randn(rng, T), 5) for _ in 1:300]
        append!(stencils, [T.((0, 0, 0, 1, 1)), T.((0, 0, 1, 1, 1)),
            T.((0, 0, 1, 0, 0)), T.((1, 1, 1, 1, 1))])
        for stencil in stencils
            p = @inferred cell_polynomial(PPM(), stencil...)
            @test p isa CellPolynomial{3,T}
            @test (@inferred center(PPM(), stencil; dx=T(0.1))) isa NamedTuple{(:value, :derivative),Tuple{T,T}}
            reference = ppm_reference(Float64.(stencil))
            reflected = cell_polynomial(PPM(), reverse(stencil)...)
            scale = max(one(T), maximum(abs, stencil))
            @test isapprox(cell_integral(p), stencil[3]; atol=tolerance * scale, rtol=tolerance)
            for x in range(-0.5, 0.5; length=9)
                v = value(p, x)
                @test isapprox(v, reference(x); atol=tolerance * scale, rtol=tolerance)
                @test isapprox(v, value(reflected, -x); atol=tolerance * scale, rtol=tolerance)
                @test min(stencil[2:4]...) - tolerance * scale <= v <= max(stencil[2:4]...) + tolerance * scale
            end
            # A quadratic is monotone throughout the cell iff its endpoint
            # derivatives have the same sign (including a flat endpoint).
            dl, dr = derivative(p, -0.5; dx=1), derivative(p, 0.5; dx=1)
            @test dl * dr >= -tolerance * scale^2
        end
        @test ppm_allocation_check(T.((0.2, 0.4, 0.8, 1.3, 1.9)), T(0.1)) == (0, 0, 0)
    end
end

@testset "PPM smooth accuracy and extremum clipping" begin
    errors = map((0.2, 0.1, 0.05)) do dx
        averages = ntuple(k -> exp((k - 3) * dx) * sinh(dx / 2) / (dx / 2), 5)
        p = cell_polynomial(PPM(), averages...)
        (abs(value(p, 0.17) - exp(0.17dx)), abs(value(p, 0.5) - exp(dx / 2)),
            abs(center(p; dx).value - 1), abs(center(p; dx).derivative - 1))
    end
    for (quantity, minimum_order) in enumerate((2.8, 3.8, 3.8, 1.8)), level in 1:2
        @test log2(errors[level][quantity] / errors[level + 1][quantity]) > minimum_order
    end
    # Classic PPM intentionally flattens a smooth extremum; no claim of
    # extremum-preserving accuracy is made for this implementation.
    averages = ntuple(k -> (k - 3)^2 + 1 // 12, 5)
    @test cell_polynomial(PPM(), averages).coefficients == (1 // 12, 0, 0)
end
