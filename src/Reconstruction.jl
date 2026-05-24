module Reconstruction

using MuladdMacro

export AbstractReconstruction,
       AbstractSlopeLimiter,
       Godunov,
       PiecewiseConstant,
       WENO3,
       WENOZ,
       MP5,
       MinmodLimiter,
       GeneralizedMinmodLimiter,
       VanLeerLimiter,
       VanAlbadaLimiter,
       MonotonizedCentralLimiter,
       SuperbeeLimiter,
       left,
       right,
       face,
       face!,
       reconstruct,
       reconstruct!,
       increment,
       slope,
       minmod,
       maxmod,
       mp5_polynomial,
       stencil_size,
       left_stencil_size,
       halo_width

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

abstract type AbstractReconstruction{H,S} end
abstract type AbstractSlopeLimiter <: AbstractReconstruction{2,4} end

"""
    Godunov()
    PiecewiseConstant()

First-order piecewise-constant reconstruction.
Has halo width `H = 1` and uses a 2-point face stencil `(u_i, u_{i+1})` and returns
`(u_i, u_{i+1})` at the `i+1/2` face.

This is the reconstruction associated with the first-order Godunov method.
It does not apply a limiter because there is no slope to limit.
"""
struct Godunov <: AbstractReconstruction{1,2} end

const PiecewiseConstant = Godunov

"""
    WENO3()

Third-order WENO reconstruction.
Has halo width `H = 2` and uses a 4-point face stencil `(u_{i-1}, u_i, u_{i+1}, u_{i+2})`.
"""
struct WENO3 <: AbstractReconstruction{2,4} end

"""
    WENOZ()

Fifth-order WENO-Z reconstruction.
Has halo width `H = 3` and uses a 6-point face stencil `(u_{i-2}, ..., u_{i+3})`.
"""
struct WENOZ <: AbstractReconstruction{3,6} end

"""
    MP5(; alpha=4.0, tolerance=0.0)

Fifth-order monotonicity-preserving reconstruction.
Has halo width `H = 3` and uses a 6-point face stencil `(u_{i-2}, ..., u_{i+3})`.
"""
struct MP5{A,T} <: AbstractReconstruction{3,6}
    alpha::A
    tolerance::T
end

MP5(; alpha=4.0, tolerance=0.0) = MP5(alpha, tolerance)

struct MinmodLimiter <: AbstractSlopeLimiter end

struct GeneralizedMinmodLimiter{T} <: AbstractSlopeLimiter
    theta::T
end

GeneralizedMinmodLimiter(; theta=1.5) = GeneralizedMinmodLimiter(theta)

struct VanLeerLimiter <: AbstractSlopeLimiter end
struct VanAlbadaLimiter <: AbstractSlopeLimiter end
struct MonotonizedCentralLimiter <: AbstractSlopeLimiter end
struct SuperbeeLimiter <: AbstractSlopeLimiter end

# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------

"""
    halo_width(recon)

Return the halo width `H` encoded in `AbstractReconstruction{H}`.

For physical cells `1:N`, this is the number of ghost cells needed
on each side to reconstruct all physical faces from `1/2` to `N+1/2`
without special boundary stencils.
"""
@inline halo_width(::AbstractReconstruction{H,S}) where {H,S} = H

"""
    stencil_size(recon)

Number of cell values needed by `face(recon, ...)`.

With the halo-width convention used here, a face reconstruction with
halo width `H` uses `2H` values around the face.
"""
@inline stencil_size(::AbstractReconstruction{H,S}) where {H,S} = S

"""
    left_stencil_size(recon)

Number of cell values needed by `left(recon, ...)`.

This is one less than the full face stencil, because the right state is
obtained by mirror symmetry from the same left-biased reconstruction.
"""
@inline left_stencil_size(::AbstractReconstruction{H,S}) where {H,S} = S-1


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@inline _tiny(x::AbstractFloat) = eps(typeof(x))
@inline _tiny(x) = eps()

@inline function minmod(a::T, b::S) where {T,S}
    return ifelse(
        a * b > 0,
        ifelse(abs(a) < abs(b), a, b),
        zero(promote_type(T, S)),
    )
end

@inline function maxmod(a::T, b::S) where {T,S}
    return ifelse(
        a * b > 0,
        ifelse(abs(a) > abs(b), a, b),
        zero(promote_type(T, S)),
    )
end

@inline function minmod(a::T, b::S, c::M) where {T,S,M}
    return ifelse(
        (a > 0) & (b > 0) & (c > 0),
        min(a, b, c),
        ifelse(
            (a < 0) & (b < 0) & (c < 0),
            max(a, b, c),
            zero(promote_type(T, S, M)),
        ),
    )
end

@inline function minmod(a::T, b::S, c::M, d::N) where {T,S,M,N}
    return ifelse(
        (a > 0) & (b > 0) & (c > 0) & (d > 0),
        min(a, b, c, d),
        ifelse(
            (a < 0) & (b < 0) & (c < 0) & (d < 0),
            max(a, b, c, d),
            zero(promote_type(T, S, M, N)),
        ),
    )
end

# ---------------------------------------------------------------------------
# First-order Godunov / piecewise-constant reconstruction
# ---------------------------------------------------------------------------

@inline left(::Godunov, u) = u

# ---------------------------------------------------------------------------
# Slope-limiter family
# ---------------------------------------------------------------------------

@inline function slope(::MinmodLimiter, u_m, u, u_p)
    Δm = u - u_m
    Δp = u_p - u
    return minmod(Δm, Δp)
end

@inline function slope(lim::GeneralizedMinmodLimiter, u_m, u, u_p)
    θ = lim.theta
    Δm = u - u_m
    Δp = u_p - u
    return minmod(θ * Δm, 0.5 * (Δm + Δp), θ * Δp)
end

@inline function slope(::VanLeerLimiter, u_m, u, u_p)
    Δm = u - u_m
    Δp = u_p - u

    return ifelse(
        Δm * Δp > 0,
        2 * Δm * Δp / (Δm + Δp),
        zero(promote_type(typeof(Δm), typeof(Δp))),
    )
end

@inline function slope(::VanAlbadaLimiter, u_m, u, u_p)
    Δm = u - u_m
    Δp = u_p - u

    return ifelse(
        Δm * Δp > 0,
        Δm * Δp * (Δm + Δp) / (Δm * Δm + Δp * Δp),
        zero(promote_type(typeof(Δm), typeof(Δp))),
    )
end

@inline function slope(::MonotonizedCentralLimiter, u_m, u, u_p)
    a = 0.5 * (u_p - u_m)
    b = 2 * (u - u_m)
    c = 2 * (u_p - u)
    return minmod(a, b, c)
end

@inline function slope(::SuperbeeLimiter, u_m, u, u_p)
    σ1 = minmod(2 * (u - u_m), u_p - u)
    σ2 = minmod(u - u_m, 2 * (u_p - u))
    return maxmod(σ1, σ2)
end

@inline function left(lim::AbstractSlopeLimiter, u_m, u, u_p)
    return u + 0.5 * slope(lim, u_m, u, u_p)
end

# ---------------------------------------------------------------------------
# WENO3
# ---------------------------------------------------------------------------


@muladd function left(::WENO3, u_m, u, u_p)
    p0 = -0.5 * u_m + 1.5 * u
    p1 =  0.5 * u   + 0.5 * u_p

    Δm = u - u_m
    Δp = u_p - u

    β0 = Δm * Δm
    β1 = Δp * Δp
    τ = (Δp - Δm) * (Δp - Δm)

    tiny = _tiny(u)

    # Optimal linear weights for third-order left reconstruction
    # at i+1/2 from points i-1, i, i+1.
    α0 = 1 // 4 * (1 + τ * inv(β0 + tiny))
    α1 = 3 // 4 * (1 + τ * inv(β1 + tiny))

    invtot = inv(α0 + α1)
    return (α0 * p0 + α1 * p1) * invtot
end

# ---------------------------------------------------------------------------
# WENO-Z
# ---------------------------------------------------------------------------

@muladd function left(::WENOZ, u_m2, u_m1, u, u_p1, u_p2)
    v1 = 3 // 8 * u_m2 - 10 // 8 * u_m1 + 15 // 8 * u
    v2 = -1 // 8 * u_m1 + 6 // 8 * u + 3 // 8 * u_p1
    v3 = 3 // 8 * u + 6 // 8 * u_p1 - 1 // 8 * u_p2

    a1 = u_m2 - 2 * u_m1 + u
    b1 = u_m2 - 4 * u_m1 + 3 * u
    a2 = u_m1 - 2 * u + u_p1
    b2 = u_m1 - u_p1
    a3 = u - 2 * u_p1 + u_p2
    b3 = 3 * u - 4 * u_p1 + u_p2

    β1 = 13 // 12 * a1 * a1 + 1 // 4 * b1 * b1
    β2 = 13 // 12 * a2 * a2 + 1 // 4 * b2 * b2
    β3 = 13 // 12 * a3 * a3 + 1 // 4 * b3 * b3

    τ5 = abs(β1 - β3)
    tiny = _tiny(v1)

    α1 = 1 // 16 * (1 + τ5 * inv(β1 + tiny))
    α2 = 5 // 8  * (1 + τ5 * inv(β2 + tiny))
    α3 = 1 // 16 * (1 + τ5 * inv(β3 + tiny))

    invtot = inv(α1 + α2 + α3)
    return (α1 * v1 + α2 * v2 + α3 * v3) * invtot
end

# ---------------------------------------------------------------------------
# MP5
# ---------------------------------------------------------------------------

@inline function mp5_polynomial(u_m2, u_m1, u, u_p1, u_p2)
    return (2 * u_m2 - 13 * u_m1 + 47 * u + 27 * u_p1 - 3 * u_p2) / 60
end

@muladd function left(lim::MP5, u_m2, u_m1, u, u_p1, u_p2)
    α = lim.alpha
    tol = lim.tolerance

    u_face = mp5_polynomial(u_m2, u_m1, u, u_p1, u_p2)
    u_mp = u + minmod(u_p1 - u, α * (u - u_m1))

    stencil_norm = sqrt(
        u_m2 * u_m2 +
        u_m1 * u_m1 +
        u * u +
        u_p1 * u_p1 +
        u_p2 * u_p2,
    )

    if (u_face - u) * (u_face - u_mp) <= tol * stencil_norm
        return u_face
    end

    d_m = u_m2 - 2 * u_m1 + u
    d_0 = u_m1 - 2 * u + u_p1
    d_p = u - 2 * u_p1 + u_p2

    dm_p = minmod(4 * d_0 - d_p, 4 * d_p - d_0, d_0, d_p)
    dm_m = minmod(4 * d_0 - d_m, 4 * d_m - d_0, d_0, d_m)

    u_ul = u + α * (u - u_m1)
    u_av = 0.5 * (u + u_p1)
    u_md = u_av - 0.5 * dm_p
    u_lc = u + 0.5 * (u - u_m1) + 4 // 3 * dm_m

    u_min = max(min(u, u_p1, u_md), min(u, u_ul, u_lc))
    u_max = min(max(u, u_p1, u_md), max(u, u_ul, u_lc))

    return u_face + minmod(u_min - u_face, u_max - u_face)
end

# ---------------------------------------------------------------------------
# Generic right, face, array versions, and increments
# ---------------------------------------------------------------------------

"""
    right(recon, stencil...)

Right-biased state obtained by mirror symmetry from `left`.
For example, `right(WENO3(), u_i, u_{i+1}, u_{i+2})`
is `left(WENO3(), u_{i+2}, u_{i+1}, u_i)`.
"""
@inline function right(recon::AbstractReconstruction, stencil::Vararg{Number,N}) where {N}
    return left(recon, reverse(stencil)...)
end

"""
    face(recon, stencil...)

Return `(left_state, right_state)` at the `i+1/2` face.

Examples:

    face(Godunov(), u_i, u_ip1)
    face(WENO3(), u_im1, u_i, u_ip1, u_ip2)
    face(WENOZ(), u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    face(MP5(),   u_im2, u_im1, u_i, u_ip1, u_ip2, u_ip3)
    face(VanLeerLimiter(), u_im1, u_i, u_ip1, u_ip2)
"""
@inline function face(
    recon::AbstractReconstruction{H,N},
    stencil::Vararg{Number,N}
) where {H,N}

    left_stencil = ntuple(j -> stencil[j], Val(N - 1))
    right_stencil = ntuple(j -> stencil[N + 1 - j], Val(N - 1))

    return (
        left(recon, left_stencil...),
        left(recon, right_stencil...),
    )
end

"""
    face!(recon, leftout, rightout, arrays...)

In-place array reconstruction. The number of arrays must match the face stencil size.
"""
function face!(
    recon::AbstractReconstruction{H,N},
    leftout::AbstractArray,
    rightout::AbstractArray,
    arrays::Vararg{AbstractArray,N},
) where {H,N}

    @inbounds for I in eachindex(leftout, rightout, arrays...)
        vals = ntuple(j -> arrays[j][I], Val(N))
        leftout[I], rightout[I] = face(recon, vals...)
    end

    return leftout, rightout
end

"""
    face(recon, arrays...)

Allocating array reconstruction.
Returns `(left, right)`.
"""
function face(recon::AbstractReconstruction{H,N}, arrays::Vararg{AbstractArray,N}) where {H,N}

    faces = face.(Ref(recon), arrays...)
    return first.(faces), last.(faces)
end

@inline function face(
    recon::AbstractReconstruction{H,N},
    stencil::NTuple{N,<:Number},
) where {H,N}
    return face(recon, stencil...)
end

@inline function face(
    recon::AbstractReconstruction{H,N},
    stencil::AbstractVector{<:Number},
) where {H,N}
    length(stencil) == N || throw(BoundsError(stencil, 1:N))
    vals = ntuple(j -> stencil[j], Val(N))
    return face(recon, vals...)
end


const reconstruct = face
const reconstruct! = face!

"""
    increment(recon, stencil...)

Return `left(recon, stencil...) - left(recon, reverse(stencil)...)`.

Useful for flux-difference-like constructions already present in the original code.
"""
@inline increment(::Godunov, u::Number) = zero(u)

@inline function increment(recon::AbstractReconstruction, stencil::Vararg{Number,N}) where {N}
    return left(recon, stencil...) - left(recon, reverse(stencil)...)
end

end # module
