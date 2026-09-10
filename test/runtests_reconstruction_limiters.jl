using Test
using Reconstruction

const RECONS = (
    Godunov(),
    WENO3(),
    WENOZ(),
    CWENO5(),
    CWENO5(; input=PointValues()),
    CWENO3(),
    CWENO3(; input=PointValues()),
    CWENO3(; weights=ZWeights()),
    CWENO3(; weights=ZWeights(), input=PointValues()),
    CWENO5(; weights=ZWeights()),
    CWENO5(; weights=ZWeights(), input=PointValues()),
    MP5(),
    MinmodLimiter(),
    GeneralizedMinmodLimiter(),
    VanLeerLimiter(),
    VanAlbadaLimiter(),
    MonotonizedCentralLimiter(),
    SuperbeeLimiter(),
)

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

    @test halo_width(CWENO5()) == 3
    @test stencil_size(CWENO5()) == 6
    @test left_stencil_size(CWENO5()) == 5
    @test halo_width(CWENO3()) == 2
    @test stencil_size(CWENO3()) == 4
    @test left_stencil_size(CWENO3()) == 3
end

@testset "Godunov / piecewise constant" begin
    @test face(Godunov(), 2.0, 3.0) == (2.0, 3.0)
    @test face(PiecewiseConstant(), 2.0, 3.0) == (2.0, 3.0)

    l = zeros(3)
    r = zeros(3)
    ui   = [1.0, 2.0, 3.0]
    uip1 = [4.0, 5.0, 6.0]

    face!(Godunov(), l, r, ui, uip1)

    @test l == ui
    @test r == uip1
end

@testset "constant-state preservation" begin
    c = 1.2345

    for recon in RECONS
        S = stencil_size(recon)
        st = ntuple(_ -> c, Val(S))

        l, r = face(recon, st...)

        @test l ≈ c atol=1e-14 rtol=1e-14
        @test r ≈ c atol=1e-14 rtol=1e-14
    end
end




@testset "linear-state reconstruction" begin
    # Use a linear stencil centered around the i+1/2 face.
    # For H=2: (-1, 0, 1, 2), exact face value is 1/2.
    # For H=3: (-2, -1, 0, 1, 2, 3), exact face value is 1/2.
    for recon in RECONS
        recon isa Godunov && continue

        H = halo_width(recon)
        S = stencil_size(recon)

        st = ntuple(k -> Float64(k - H), Val(S))
        l, r = face(recon, st...)

        @test l ≈ 0.5 atol=1e-12 rtol=1e-12
        @test r ≈ 0.5 atol=1e-12 rtol=1e-12
    end
end

@testset "WENO-Z point interpolation accuracy" begin
    # Symmetry makes tau5 zero for this cubic, exposing the optimal weights.
    cubic = (-8.0, -1.0, 0.0, 1.0, 8.0)
    @test left(WENOZ(), cubic...) ≈ 1 / 8 atol=1e-14
    @test right(WENOZ(), cubic...) ≈ -1 / 8 atol=1e-14

    # Check both face states converge at fifth order on smooth point data.
    errors = map((0.2, 0.1, 0.05)) do dx
        stencil = ntuple(k -> exp((k - 3) * dx), 6)
        l, r = face(WENOZ(), stencil...)
        (abs(l - exp(dx / 2)), abs(r - exp(dx / 2)))
    end
    for side in 1:2, level in 1:2
        order = log2(errors[level][side] / errors[level + 1][side])
        @test 4.7 < order < 5.3
    end
end

st4 = (-1.0, 0.0, 1.0, 2.0)
    st6 = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)

    @test face(WENO3(), st4) == face(WENO3(), st4...)
    @test face(WENOZ(), st6) == face(WENOZ(), st6...)
    @test face(WENO3(), collect(st4)) == face(WENO3(), st4...)
    @test face(WENOZ(), collect(st6)) == face(WENOZ(), st6...)
    @test_throws BoundsError face(WENOZ(), collect(st4))



@testset "tuple and vector stencil convenience" begin
    st4 = (-1.0, 0.0, 1.0, 2.0)
    st6 = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)

    @test face(WENO3(), st4) == face(WENO3(), st4...)
    @test face(WENOZ(), st6) == face(WENOZ(), st6...)

    # This assumes the AbstractVector method is written as:
    #
    #     face(recon::AbstractReconstruction{H,S}, stencil::AbstractVector{<:Number}) where {H,S}
    #
    # and checks length(stencil) == S before converting to ntuple.
    @test face(WENO3(), collect(st4)) == face(WENO3(), st4...)
    @test face(WENOZ(), collect(st6)) == face(WENOZ(), st6...)

    @test_throws BoundsError face(WENOZ(), collect(st4))
end

@testset "in-place array reconstruction" begin
    # Two independent faces with linear data.
    um  = [-1.0, 1.0]
    u   = [ 0.0, 2.0]
    up  = [ 1.0, 3.0]
    up2 = [ 2.0, 4.0]

    l = zeros(2)
    r = zeros(2)

    face!(VanLeerLimiter(), l, r, um, u, up, up2)

    @test l ≈ [0.5, 2.5]
    @test r ≈ [0.5, 2.5]
end

@testset "wrong scalar stencil size does not dispatch" begin
    @test_throws MethodError face(WENO3(), 1.0, 2.0, 3.0)
    @test_throws MethodError face(WENOZ(), 1.0, 2.0, 3.0, 4.0)

    l = zeros(2)
    r = zeros(2)
    a = ones(2)

    @test_throws MethodError face!(WENOZ(), l, r, a, a, a, a)
    @test_throws MethodError face!(WENO3(), l, r, a, a, a, a,a,a)
end

include(joinpath(@__DIR__, "runtests_cell_polynomials.jl"))
