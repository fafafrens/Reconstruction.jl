using Reconstruction

const RECONSTRUCTIONS = (
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

const PLOT_RECONS = filter(r -> !(r isa Union{GeneralizedMinmodLimiter,VanAlbadaLimiter,MonotonizedCentralLimiter}), RECONSTRUCTIONS)

has_center(recon) = recon isa Union{Godunov,AbstractSlopeLimiter,WENO3,WENOZ,CWENO3,CWENO5}

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
    H = halo_width(recon)
    S = stencil_size(recon)

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
