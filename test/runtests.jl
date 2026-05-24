using Test
using Reconstruction

const ALL_RECONS = (
    Godunov(),
    WENO3(),
    WENOZ(),
    MP5(),
    MinmodLimiter(),
    GeneralizedMinmodLimiter(),
    VanLeerLimiter(),
    VanAlbadaLimiter(),
    MonotonizedCentralLimiter(),
    SuperbeeLimiter(),
)

const NON_GODUNOV_RECONS = (
    WENO3(),
    WENOZ(),
    MP5(),
    MinmodLimiter(),
    GeneralizedMinmodLimiter(),
    VanLeerLimiter(),
    VanAlbadaLimiter(),
    MonotonizedCentralLimiter(),
    SuperbeeLimiter(),
)

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

@testset "traits" begin
    @test halo_width(Godunov()) == 1
    @test stencil_size(Godunov()) == 2
    @test left_stencil_size(Godunov()) == 1

    @test halo_width(WENO3()) == 2
    @test stencil_size(WENO3()) == 4
    @test left_stencil_size(WENO3()) == 3

    @test halo_width(WENOZ()) == 3
    @test stencil_size(WENOZ()) == 6
    @test left_stencil_size(WENOZ()) == 5

    @test halo_width(MP5()) == 3
    @test stencil_size(MP5()) == 6
    @test left_stencil_size(MP5()) == 5
end

@testset "small helpers" begin
    @test minmod(1, 2) == 1
    @test minmod(2, 1) == 1
    @test minmod(-2, -1) == -1
    @test minmod(-1, 2) == 0
    @test minmod(1, 2, 3) == 1
    @test minmod(-1, -2, -3) == -1
    @test minmod(1, -2, 3) == 0
    @test minmod(1, 2, 3, 4) == 1
    @test minmod(-1, -2, -3, -4) == -1
    @test minmod(1, 2, -3, 4) == 0

    @test maxmod(1, 2) == 2
    @test maxmod(2, 1) == 2
    @test maxmod(-2, -1) == -2
    @test maxmod(-1, 2) == 0
end

@testset "Godunov / piecewise constant" begin
    @test face(Godunov(), 2.0, 3.0) == (2.0, 3.0)
    @test face(PiecewiseConstant(), 2.0, 3.0) == (2.0, 3.0)
    @test increment(Godunov(), 2.0) == 0.0

    left = zeros(3)
    right = zeros(3)
    u_i = [1.0, 2.0, 3.0]
    u_ip1 = [4.0, 5.0, 6.0]

    returned_left, returned_right = face!(Godunov(), left, right, u_i, u_ip1)
    @test returned_left === left
    @test returned_right === right
    @test left == u_i
    @test right == u_ip1
end

@testset "constant-state preservation" begin
    for recon in ALL_RECONS
        s = stencil_size(recon)
        stencil = ntuple(_ -> 1.2345, Val(s))
        left, right = face(recon, stencil...)

        @test left ≈ 1.2345 atol = 1.0e-14 rtol = 1.0e-14
        @test right ≈ 1.2345 atol = 1.0e-14 rtol = 1.0e-14
    end
end

@testset "linear-state reconstruction" begin
    for recon in NON_GODUNOV_RECONS
        h = halo_width(recon)
        s = stencil_size(recon)
        stencil = ntuple(k -> Float64(k - h), Val(s))
        left, right = face(recon, stencil...)

        @test left ≈ 0.5 atol = 1.0e-12 rtol = 1.0e-12
        @test right ≈ 0.5 atol = 1.0e-12 rtol = 1.0e-12
    end
end

@testset "tuple and vector stencil convenience" begin
    st4 = (-1.0, 0.0, 1.0, 2.0)
    st6 = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)

    @test face(WENO3(), st4) == face(WENO3(), st4...)
    @test face(WENOZ(), st6) == face(WENOZ(), st6...)
    @test face(WENO3(), collect(st4)) == face(WENO3(), st4...)
    @test face(WENOZ(), collect(st6)) == face(WENOZ(), st6...)
    @test_throws BoundsError face(WENOZ(), collect(st4))
end

@testset "in-place array reconstruction" begin
    u_m = [-1.0, 1.0]
    u = [0.0, 2.0]
    u_p = [1.0, 3.0]
    u_p2 = [2.0, 4.0]

    left = zeros(2)
    right = zeros(2)
    returned_left, returned_right = face!(VanLeerLimiter(), left, right, u_m, u, u_p, u_p2)

    @test returned_left === left
    @test returned_right === right
    @test left ≈ [0.5, 2.5]
    @test right ≈ [0.5, 2.5]
end

@testset "allocating array reconstruction" begin
    stencil = (
        [-1.0, 1.0],
        [0.0, 2.0],
        [1.0, 3.0],
        [2.0, 4.0],
    )

    left, right = face(MonotonizedCentralLimiter(), stencil...)
    @test left ≈ [0.5, 2.5]
    @test right ≈ [0.5, 2.5]
end

@testset "slopes" begin
    @test slope(GeneralizedMinmodLimiter(theta=1.0), 1.0, 2.0, 4.0) ≈ 1.0
    @test slope(GeneralizedMinmodLimiter(), 1.0, 2.0, 4.0) ≈ 1.5
    @test slope(GeneralizedMinmodLimiter(theta=2.0), 1.0, 2.0, 4.0) ≈ 1.5

    @test slope(VanLeerLimiter(), 1.0, 2.0, 4.0) ≈ 4 / 3
    @test slope(VanAlbadaLimiter(), 1.0, 2.0, 4.0) ≈ 6 / 5

    for limiter in (GeneralizedMinmodLimiter(), VanLeerLimiter(), VanAlbadaLimiter())
        @test slope(limiter, 1.0, 2.0, 1.0) == 0.0
        @test slope(limiter, 3.0, 2.0, 3.0) == 0.0
    end
end

@testset "MP5 limiting and increments" begin
    @test face(MP5(), 0.0, 0.0, 0.0, 10.0, 10.0, 10.0) == (0.0, 10.0)
    @test left(MP5(), 10.0, 10.0, 10.0, 0.0, 0.0) == 10.0
    @test mp5_polynomial(0.0, 1.0, 2.0, 3.0, 4.0) ≈ 2.5
    @test increment(MP5(), 0.0, 1.0, 2.0, 3.0, 4.0) ≈ 1.0
    @test increment(WENOZ(), 0.0, 1.0, 2.0, 3.0, 4.0) ≈ 1.0
    @test increment(WENO3(), 1.0, 2.0, 3.0) ≈ 1.0
end

@testset "wrong stencil sizes" begin
    @test_throws MethodError face(WENO3(), 1.0, 2.0, 3.0)
    @test_throws MethodError face(WENOZ(), 1.0, 2.0, 3.0, 4.0)

    left = zeros(2)
    right = zeros(2)
    a = ones(2)

    @test_throws MethodError face!(WENOZ(), left, right, a, a, a, a)
    @test_throws MethodError face!(WENO3(), left, right, a, a, a, a, a, a)
end

@testset "periodic reconstruction smoke test" begin
    n = 64
    x = ((0:n - 1) .+ 0.5) ./ n
    profiles = (
        x -> sin(2π * x),
        x -> x < 0.5 ? 1.0 : 0.0,
        x -> max(0.0, 1.0 - abs(x - 0.5) / 0.2),
    )

    for f in profiles
        u = f.(x)
        for recon in ALL_RECONS
            left, right = reconstruct_periodic_faces(recon, u)
            @test length(left) == n
            @test length(right) == n
            @test all(isfinite, left)
            @test all(isfinite, right)
        end
    end
end
