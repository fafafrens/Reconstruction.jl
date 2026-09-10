# test_and_plot_reconstruction_limiters.jl
#
# Run from the package root:
#
#     julia --project=. examples/test_and_plot_reconstruction_limiters.jl
#
# It runs small correctness tests and writes diagnostic plots to:
#
#     examples/reconstruction_plots/
#
# Requirements for the plotting part:
#
#     import Pkg; Pkg.add("Plots")
#
# The tests need Test and the reconstruction module's dependencies.

using Test

using Reconstruction

# ---------------------------------------------------------------------------
# Small compatibility helper
# ---------------------------------------------------------------------------
# Works whether traits return integers, e.g. stencil_size(WENOZ()) == 6,
# or Val objects, e.g. stencil_size(WENOZ()) == Val(6).

_int(x::Integer) = x
_int(::Val{N}) where {N} = N

# ---------------------------------------------------------------------------
# Reconstruction lists
# ---------------------------------------------------------------------------

const ALL_RECONS = (
    Godunov(),
    CWENO3(; weights=ZWeights()),
    CWENO3(; weights=ZWeights(), input=PointValues()),
    CWENO5(; weights=ZWeights()),
    CWENO5(; weights=ZWeights(), input=PointValues()),
    CWENO3(),
    CWENO3(; input=PointValues()),
    WENO3(),
    WENOZ(),
    CWENO5(),
    CWENO5(; input=PointValues()),
    MP5(),
    MinmodLimiter(),
    GeneralizedMinmodLimiter(),
    VanLeerLimiter(),
    VanAlbadaLimiter(),
    MonotonizedCentralLimiter(),
    SuperbeeLimiter(),
)

const NON_GODUNOV_RECONS = (
    CWENO3(; weights=ZWeights()),
    CWENO3(; weights=ZWeights(), input=PointValues()),
    CWENO5(; weights=ZWeights()),
    CWENO5(; weights=ZWeights(), input=PointValues()),
    CWENO3(),
    CWENO3(; input=PointValues()),
    WENO3(),
    WENOZ(),
    CWENO5(),
    CWENO5(; input=PointValues()),
    MP5(),
    MinmodLimiter(),
    GeneralizedMinmodLimiter(),
    VanLeerLimiter(),
    VanAlbadaLimiter(),
    MonotonizedCentralLimiter(),
    SuperbeeLimiter(),
)

const PLOT_RECONS = (
    Godunov(),
    CWENO3(; weights=ZWeights()),
    CWENO3(; weights=ZWeights(), input=PointValues()),
    CWENO5(; weights=ZWeights()),
    CWENO5(; weights=ZWeights(), input=PointValues()),
    CWENO3(),
    CWENO3(; input=PointValues()),
    MinmodLimiter(),
    VanLeerLimiter(),
    SuperbeeLimiter(),
    WENO3(),
    WENOZ(),
    CWENO5(),
    CWENO5(; input=PointValues()),
    MP5(),
)

recon_label(recon) = string(nameof(typeof(recon)))
recon_label(::MP5) = "MP5"
recon_label(r::Union{CWENO3,CWENO5}) = string(nameof(typeof(r)), r.weights isa ZWeights ? "-Z" : "",
    r.input isa CellAverages ? " (averages)" : " (points)")

# Exact primitives let the finite-volume reconstructions receive cell averages,
# including cells that straddle the discontinuities in these profiles.
const PROFILES = (
    ("sine", x -> sin(2π * x), x -> -cos(2π * x) / (2π)),
    ("step", x -> x < 0.5 ? 1.0 : 0.0, x -> min(x, 0.5)),
    ("hat", x -> max(0.0, 1.0 - abs(x - 0.5) / 0.2),
        x -> (max(x - 0.3, 0)^2 - 2max(x - 0.5, 0)^2 + max(x - 0.7, 0)^2) / 0.4),
    ("mixed", x -> sin(2π * x) + (0.35 <= x <= 0.65 ? 0.75 : 0.0),
        x -> -cos(2π * x) / (2π) + 0.75 * max(0, min(x, 0.65) - 0.35)),
)

function sample_profile(recon, f, primitive, N)
    averages = recon isa Union{Godunov,AbstractSlopeLimiter,MP5} ||
        (recon isa Union{CWENO3,CWENO5} && recon.input isa CellAverages)
    return averages ? [N * (primitive(i / N) - primitive((i - 1) / N)) for i in 1:N] :
        [f((i - 0.5) / N) for i in 1:N]
end

# ---------------------------------------------------------------------------
# Utility: reconstruct all periodic faces
# ---------------------------------------------------------------------------
# Given cell values u[1:N], reconstruct states at faces i+1/2, i = 1,...,N,
# with periodic indexing. The face stencil is
#
#     u_{i-H+1}, ..., u_{i+H}
#
# where H = halo_width(recon), S = stencil_size(recon) = 2H for the current
# centered reconstructions.

function reconstruct_periodic_faces(recon, u::AbstractVector)
    Ncells = length(u)
    H = _int(halo_width(recon))
    S = _int(stencil_size(recon))

    left_states = Vector{float(eltype(u))}(undef, Ncells)
    right_states = similar(left_states)

    @inbounds for i in 1:Ncells
        stencil = ntuple(j -> u[mod1(i - H + j, Ncells)], Val(S))
        left_states[i], right_states[i] = face(recon, stencil...)
    end

    return left_states, right_states
end

function reconstruct_periodic_centers(recon, u::AbstractVector)
    N = length(u)
    H = halo_width(recon)
    S = left_stencil_size(recon)
    values, derivatives = similar(u), similar(u)
    for i in eachindex(u)
        stencil = ntuple(j -> u[mod1(i - H + j, N)], Val(S))
        values[i], derivatives[i] = center(recon, stencil; dx=1 / N)
    end
    return values, derivatives
end

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@testset "Godunov / piecewise constant" begin
    @test face(Godunov(), 2.0, 3.0) == (2.0, 3.0)
    @test face(PiecewiseConstant(), 2.0, 3.0) == (2.0, 3.0)

    l = zeros(3)
    r = zeros(3)
    ui = [1.0, 2.0, 3.0]
    uip1 = [4.0, 5.0, 6.0]

    face!(Godunov(), l, r, ui, uip1)

    @test l == ui
    @test r == uip1
end

@testset "constant-state preservation" begin
    c = 1.2345

    for recon in ALL_RECONS
        S = _int(stencil_size(recon))
        st = ntuple(_ -> c, Val(S))

        l, r = face(recon, st...)

        @test l ≈ c atol=1e-14 rtol=1e-14
        @test r ≈ c atol=1e-14 rtol=1e-14
    end
end

@testset "linear-state reconstruction" begin
    # For a linear stencil centered around the face i+1/2:
    #
    # H = 2: (-1, 0, 1, 2)
    # H = 3: (-2, -1, 0, 1, 2, 3)
    #
    # The exact face value is 1/2.
    #
    # Godunov is first-order and is not expected to pass this test.

    for recon in NON_GODUNOV_RECONS
        H = _int(halo_width(recon))
        S = _int(stencil_size(recon))

        st = ntuple(k -> Float64(k - H), Val(S))
        l, r = face(recon, st...)

        @test l ≈ 0.5 atol=1e-12 rtol=1e-12
        @test r ≈ 0.5 atol=1e-12 rtol=1e-12
    end
end

@testset "tuple stencil convenience" begin
    st4 = (-1.0, 0.0, 1.0, 2.0)
    st6 = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)

    @test face(WENO3(), st4) == face(WENO3(), st4...)
    @test face(WENOZ(), st6) == face(WENOZ(), st6...)
end

@testset "vector stencil convenience" begin
    # These tests assume you implemented the AbstractVector method as:
    #
    # face(recon::AbstractReconstruction{H,S}, stencil::AbstractVector{<:Number}) where {H,S}
    #
    # with an explicit length check that throws ArgumentError.

    st4 = (-1.0, 0.0, 1.0, 2.0)
    st6 = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)

    @test face(WENO3(), collect(st4)) == face(WENO3(), st4...)
    @test face(WENOZ(), collect(st6)) == face(WENOZ(), st6...)

    @test_throws BoundsError face(WENOZ(), collect(st4))
end

@testset "in-place array reconstruction" begin
    # Two independent faces with linear data.
    um = [-1.0, 1.0]
    u = [0.0, 2.0]
    up = [1.0, 3.0]
    up2 = [2.0, 4.0]

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
    @test_throws MethodError face!(WENO3(), l, r, a, a, a, a, a, a)
end

@testset "periodic reconstruction smoke test" begin
    N = 64
    for (_, f, primitive) in PROFILES
        for recon in ALL_RECONS
            u = sample_profile(recon, f, primitive, N)
            l, r = reconstruct_periodic_faces(recon, u)

            @test length(l) == N
            @test length(r) == N
            @test all(isfinite, l)
            @test all(isfinite, r)
            if recon isa Union{Godunov,AbstractSlopeLimiter,CWENO3,CWENO5}
                c, d = reconstruct_periodic_centers(recon, u)
                @test length(c) == N
                @test length(d) == N
                @test all(isfinite, c)
                @test all(isfinite, d)
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Plotting diagnostics
# ---------------------------------------------------------------------------
# This script writes figures to files and does not need a GUI window.
get!(ENV, "GKSwstype", "100")
const HAS_PLOTS = try
    @eval using Plots: plot, plot!, savefig
    true
catch err
    @warn "Plots.jl is not available; skipping plotting diagnostics." exception=(err, catch_backtrace())
    false
end

function make_reconstruction_plots(; N=96, outdir=joinpath(@__DIR__, "reconstruction_plots"))
    HAS_PLOTS || return String[]

    mkpath(outdir)

    xcell = ((0:N-1) .+ 0.5) ./ N
    xface = (1:N) ./ N

    written = String[]

    for (name, f, primitive) in PROFILES

        p = plot(
            xcell,
            f.(xcell);
            label="exact point values",
            linewidth=3,
            xlabel="x",
            ylabel="reconstructed left state",
            title="Left state at faces: $name",
            legend=:outerright,
        )

        for recon in PLOT_RECONS
            u = sample_profile(recon, f, primitive, N)
            l, _ = reconstruct_periodic_faces(recon, u)
            plot!(p, xface, l; label=recon_label(recon), linewidth=1.5)
        end

        path_left = joinpath(outdir, "$(name)_left_states.png")
        savefig(p, path_left)
        push!(written, path_left)

        q = plot(
            xface,
            zeros(N);
            label="zero",
            linewidth=2,
            xlabel="x",
            ylabel="|uR - uL|",
            title="Face jump diagnostic: $name",
            legend=:outerright,
        )

        for recon in PLOT_RECONS
            u = sample_profile(recon, f, primitive, N)
            l, r = reconstruct_periodic_faces(recon, u)
            plot!(q, xface, abs.(r .- l); label=recon_label(recon), linewidth=1.5)
        end

        path_jump = joinpath(outdir, "$(name)_face_jumps.png")
        savefig(q, path_jump)
        push!(written, path_jump)

        # A smooth profile provides an exact derivative for comparison.
        if name == "sine"
            cplot = plot(xcell, f.(xcell); label="exact", xlabel="x",
                ylabel="center value", title="Cell-center values: sine", legend=:outerright)
            dplot = plot(xcell, 2π .* cos.(2π .* xcell); label="exact", xlabel="x",
                ylabel="du/dx", title="Cell-center derivatives: sine", legend=:outerright)
            for recon in PLOT_RECONS
                recon isa Union{Godunov,AbstractSlopeLimiter,CWENO3,CWENO5} || continue
                u = sample_profile(recon, f, primitive, N)
                c, d = reconstruct_periodic_centers(recon, u)
                plot!(cplot, xcell, c; label=recon_label(recon))
                plot!(dplot, xcell, d; label=recon_label(recon))
            end
            for (figure, suffix) in ((cplot, "center_values"), (dplot, "center_derivatives"))
                path = joinpath(outdir, "sine_$(suffix).png")
                savefig(figure, path)
                push!(written, path)
            end
        end
    end

    return written
end

if abspath(PROGRAM_FILE) == @__FILE__
    written = make_reconstruction_plots()
    if !isempty(written)
        @info "Wrote reconstruction diagnostic plots" written
    end
end
