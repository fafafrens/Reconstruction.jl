# CWENO types, constructor-time tables, and specialized reconstruction kernels.

"""Input samples represent point values at uniformly spaced cell centers."""
struct PointValues end

"""Input samples represent averages over uniform cells."""
struct CellAverages end

"""Ordinary CWENO Jiang-Shu weights with exponent two (the default)."""
struct JSWeights end

"""
    ZWeights(; power=2)

CWENO-Z weights `α_k = d_k * (1 + (τ / (β_k + epsilon))^power)`.
Supports `power=1` or `power=2`. The central indicator measures the optimal
polynomial, while the additional candidate P₀ is still used in the blend.
The global indicator is specific to the CWENO order and input convention.
"""
struct ZWeights{P}
    function ZWeights{P}() where {P}
        P === 1 || P === 2 || throw(ArgumentError("ZWeights power must be 1 or 2"))
        return new{P}()
    end
end
ZWeights(; power::Integer=2) = ZWeights{Int(power)}()

# Fitting and integration happen when a reconstruction object is constructed.
# Exact rational tables avoid rounding the coefficients during their derivation.
_cell_moment(k) = isodd(k) ? 0 // 1 : 1 // ((k + 1) * 2^k)
_sample_moment(::PointValues, j, k) = (j // 1)^k
_sample_moment(::CellAverages, j, k) =
    ((j + 1 // 2)^(k + 1) - (j - 1 // 2)^(k + 1)) / (k + 1)

function _exact_fit(input, offsets)
    n = length(offsets)
    A = [_sample_moment(input, j, k) for j in offsets, k in 0:(n - 1)]
    M = inv(A)
    return ntuple(i -> ntuple(j -> M[i, j], n), n)
end

_exact_fits(input, ::Val{5}) = (
    optimal=_exact_fit(input, -2:2),
    left=_exact_fit(input, -2:0),
    center=_exact_fit(input, -1:1),
    right=_exact_fit(input, 0:2),
)

function _exact_smoothness(::Val{N}) where {N}
    return ntuple(N) do i
        ntuple(N) do j
            k, m = i - 1, j - 1
            sum((factorial(k) ÷ factorial(k - l)) *
                (factorial(m) ÷ factorial(m - l)) * _cell_moment(k + m - 2l)
                for l in 1:min(k, m); init=0 // 1)
        end
    end
end

_exact_fits(input, ::Val{3}) = (
    optimal=_exact_fit(input, -1:1),
    left=_exact_fit(input, -1:0),
    right=_exact_fit(input, 0:1),
)

# Derive τ as a quadratic form in the optimal polynomial's coefficients.
# Combining the indicators exactly here removes their common low-order terms
# before floating-point evaluation (rather than subtracting nearly equal βs).
_tau_factors(::Union{PointValues,CellAverages}, ::Val{3}) = (1, 1)
_tau_factors(::CellAverages, ::Val{5}) = (1, 4, 1)
_tau_factors(::PointValues, ::Val{5}) = (1, 6, 1)
_table_matrix(table) = reduce(vcat, map(row -> permutedims(collect(row)), table))

function _exact_tau(input, order::Val{N}, fits, smoothness) where {N}
    B = _table_matrix(smoothness)
    stencils = N == 3 ? (-1:0, 0:1) : (-2:0, -1:1, 0:2)
    factors = _tau_factors(input, order)
    Q = -sum(factors) * B
    for (stencil, factor, fit) in zip(stencils, factors, Base.tail(Tuple(fits)))
        n = length(stencil)
        A = [_sample_moment(input, j, k) for j in stencil, k in 0:(N - 1)]
        projection = _table_matrix(fit) * A
        Q += factor * transpose(projection) * B[1:n, 1:n] * projection
    end
    # Capture a fresh, concretely typed binding after the accumulation loop.
    # This keeps the constructor's returned table type inferable.
    matrix = Q::Matrix{Rational{Int}}
    return ntuple(i -> ntuple(j -> matrix[i, j], Val(N)), Val(N))
end

struct CWENOCoefficients{F,B,Q}
    fits::F
    smoothness::B
    tau::Q
end

_convert_table(::Type{T}, table) where {T} = map(row -> map(T, row), table)

function _cweno_coefficients(input, order, ::Type{T}, weights) where {T}
    fits = _exact_fits(input, order)
    smoothness = _exact_smoothness(order)
    tau = weights isa ZWeights ?
        _convert_table(T, _exact_tau(input, order, fits, smoothness)) : nothing
    return CWENOCoefficients(map(table -> _convert_table(T, table), fits),
        _convert_table(T, smoothness), tau)
end

"""
    CWENO3(; input=CellAverages(), epsilon=1e-6, weights=JSWeights())

Third-order CWENO reconstruction on a uniform grid. Produces a quadratic from
three centered samples; face reconstruction uses four samples and a two-cell halo.
Supports `PointValues()` and `CellAverages()` with immutable fitting and
Jiang-Shu smoothness tables built by the constructor and stored in `coefficients`.

Uses an optimal quadratic, two linear candidates, and an additional quadratic
candidate, with linear weights `(1/2, 1/4, 1/4)`. The default JSWeights uses
nonlinear exponent two; select `weights=ZWeights()` for CWENO-Z.
Smooth values target third order and first derivatives second order. The type
and scaling of the positive `epsilon` parameter follow the CWENO5 convention;
use `epsilon=1f-6` for Float32 tables. Construct once outside the cell loop.

Neither weighting strategy guarantees TVD or positivity.
Reference: Cravero et al., https://arxiv.org/abs/1607.07319, section 3.
"""
struct CWENO3{D,T<:AbstractFloat,C,W} <: AbstractReconstruction{2,4}
    input::D
    epsilon::T
    coefficients::C
    weights::W
end

function CWENO3(; input::Union{PointValues,CellAverages}=CellAverages(), epsilon::Real=1e-6,
    weights::Union{JSWeights,ZWeights}=JSWeights())
    ε = float(epsilon)
    isfinite(ε) && ε > 0 || throw(ArgumentError("epsilon must be finite and positive"))
    coefficients = _cweno_coefficients(input, Val(3), typeof(ε), weights)
    return CWENO3(input, ε, coefficients, weights)
end

"""
    CWENO5(; input=CellAverages(), epsilon=1e-6, weights=JSWeights())

Fifth-order CWENO reconstruction on a uniform grid. Produces one quartic per
cell from five centered samples. Face reconstruction uses six samples and a
three-cell halo, as with WENOZ and MP5.

`input` explicitly selects point values or cell averages. Fixed fitting and
Jiang-Shu smoothness tables are built by the constructor and stored as immutable
tuples in `coefficients`.
The floating-point type of `epsilon` selects their numeric type, e.g. `1f-6`
for Float32. Construct this object once outside the reconstruction loop.

Uses linear weights `(1/2, 1/6, 1/6, 1/6)` and, by default, Jiang-Shu nonlinear
weights with exponent two. Select `weights=ZWeights()` for CWENO-Z.
`epsilon` must be finite and positive and has
the units of the squared reconstructed variable. Its choice affects resolution
and accuracy near critical points; the default is intended for data of order one.
Mesh-dependent choices can be supplied explicitly, e.g. `epsilon=dx^2` for
nondimensional data. A smooth value reconstruction targets fifth order; its
first derivative targets fourth order. No TVD or positivity guarantee is implied.

Reference: Cravero et al., https://arxiv.org/abs/1607.07319, section 3.
"""
struct CWENO5{D,T<:AbstractFloat,C,W} <: AbstractReconstruction{3,6}
    input::D
    epsilon::T
    coefficients::C
    weights::W
end

function CWENO5(; input::Union{PointValues,CellAverages}=CellAverages(), epsilon::Real=1e-6,
    weights::Union{JSWeights,ZWeights}=JSWeights())
    ε = float(epsilon)
    isfinite(ε) && ε > 0 || throw(ArgumentError("epsilon must be finite and positive"))
    coefficients = _cweno_coefficients(input, Val(5), typeof(ε), weights)
    return CWENO5(input, ε, coefficients, weights)
end

@inline _tuple_dot(a, b) = sum(map(*, a, b))
@inline _fit(table, u) = map(row -> _tuple_dot(row, u), table)
@inline _smoothness(table, a) = max(zero(a[1]), _tuple_dot(a, _fit(table, a)))

# Jiang-Shu matrices have fixed sparsity in the monomial basis. Keep the
# coefficients in the reconstruction object, but skip their structural zeros.
@inline @muladd function _smoothness(table::NTuple{3}, a::NTuple{3})
    β = table[2][2] * a[2]^2 + table[3][3] * a[3]^2
    return max(zero(β), β)
end

@inline @muladd function _smoothness(table::NTuple{5}, a::NTuple{5})
    β = a[2] * (table[2][2] * a[2] + 2 * table[2][4] * a[4]) +
        a[3] * (table[3][3] * a[3] + 2 * table[3][5] * a[5]) +
        table[4][4] * a[4]^2 + table[5][5] * a[5]^2
    return max(zero(β), β)
end

@inline _cweno_weights(recon, optimal, candidates, d) =
    _cweno_weights(recon.weights, recon, optimal, candidates, d)

@inline function _cweno_weights(::JSWeights, recon, optimal, candidates, d::NTuple{N}) where {N}
    β = map(a -> _smoothness(recon.coefficients.smoothness, a), candidates)
    denominators = map(b -> b + recon.epsilon, β)
    scale = minimum(denominators)
    # Common rescaling is algebraically identical to d/(β+epsilon)^2,
    # while avoiding overflow when epsilon is very small.
    α = ntuple(k -> d[k] * (scale / denominators[k])^2, Val(N))
    invtotal = inv(sum(α))
    return map(a -> a * invtotal, α)
end

@inline _cweno_tau(recon, optimal) =
    abs(_tuple_dot(optimal, _fit(recon.coefficients.tau, optimal)))

@inline function _cweno_tau(recon::CWENO3, a::NTuple{3})
    return abs(recon.coefficients.tau[3][3] * a[3]^2)
end

@inline @muladd function _cweno_tau(recon::CWENO5, a::NTuple{5})
    Q = recon.coefficients.tau
    return abs(Q[4][4] * a[4]^2 + a[5] * (2 * Q[3][5] * a[3] + Q[5][5] * a[5]))
end

@inline _weight_power(::ZWeights{1}, x) = x
@inline _weight_power(::ZWeights{2}, x) = x * x

@inline function _z_normalized_weights(weights::ZWeights, β, τ, epsilon, d::NTuple{N}) where {N}
    denominators = map(b -> b + epsilon, β)
    smallest = minimum(denominators)
    scale = max(smallest, τ)
    # Divide all α by (scale/smallest)^P. All ratios are <= 1, even when
    # τ/(β+epsilon) would overflow. τ=0 recovers the linear weights.
    base = _weight_power(weights, smallest / scale)
    # τ*smallest/scale equals min(τ, smallest). This removes a division and
    # the per-candidate products, while preserving the bounded-ratio scaling.
    scaled_tau = min(τ, smallest)
    α = ntuple(k -> d[k] * (base + _weight_power(weights, scaled_tau / denominators[k])), Val(N))
    invtotal = inv(sum(α))
    return map(a -> a * invtotal, α)
end

@inline function _cweno_weights(weights::ZWeights, recon, optimal, candidates, d::NTuple{N}) where {N}
    β = ntuple(k -> _smoothness(recon.coefficients.smoothness,
        k == 1 ? optimal : candidates[k]), Val(N))
    return _z_normalized_weights(weights, β, _cweno_tau(recon, optimal), recon.epsilon, d)
end

@inline _constant_coefficient(::PointValues, u, a) = u + zero(a[1])
@inline _constant_coefficient(::CellAverages, u, a) = u - a[3] / 12 - a[5] / 80
@inline _constant_coefficient(::CellAverages, u, a::NTuple{3}) = u - a[3] / 12

# The centered data have v[center] == 0. Parity of the optimal fitting matrix
# pairs symmetric samples, avoiding dense products and repeated differences.
# Constant coefficients are unnecessary for smoothness and are imposed after
# blending to satisfy the central interpolation/conservation constraint.
@inline @muladd function _centered_optimal(table::NTuple{3}, v::NTuple{3})
    a1 = table[2][3] * (v[3] - v[1])
    a2 = table[3][1] * (v[1] + v[3])
    return (zero(a1), a1, a2)
end

@inline @muladd function _centered_optimal(table::NTuple{5}, v::NTuple{5})
    even_near, even_far = v[2] + v[4], v[1] + v[5]
    odd_near, odd_far = v[4] - v[2], v[5] - v[1]
    a1 = table[2][4] * odd_near + table[2][5] * odd_far
    a2 = table[3][2] * even_near + table[3][1] * even_far
    a3 = table[4][4] * odd_near + table[4][5] * odd_far
    a4 = table[5][2] * even_near + table[5][1] * even_far
    return (zero(a1), a1, a2, a3, a4)
end

@inline @muladd function cell_polynomial(recon::CWENO3, um::Real, u::Real, up::Real)
    v = (um - u, zero(u), up - u)
    fits = recon.coefficients.fits
    optimal = _centered_optimal(fits.optimal, v)
    al = fits.left[2][1] * v[1]
    ar = fits.right[2][2] * v[3]
    pl = (zero(al), al, zero(al))
    pr = (zero(ar), ar, zero(ar))
    p0 = ntuple(j -> 2 * optimal[j] - (pl[j] + pr[j]) / 2, Val(3))
    candidates = (p0, pl, pr)
    half = one(recon.epsilon) / 2
    quarter = one(recon.epsilon) / 4
    ω = _cweno_weights(recon, optimal, candidates, (half, quarter, quarter))
    a1 = ω[1] * p0[2] + ω[2] * al + ω[3] * ar
    a2 = ω[1] * p0[3]
    a = (zero(a1), a1, a2)
    a0 = _constant_coefficient(recon.input, u, a)
    return CellPolynomial((a0, a[2], a[3]))
end

@inline @muladd function cell_polynomial(recon::CWENO5, um2::Real, um1::Real, u::Real, up1::Real, up2::Real)
    # Subtract the central sample to preserve constants and avoid cancellation
    # of a large common offset when fitting slopes and curvatures.
    v = (um2 - u, um1 - u, zero(u), up1 - u, up2 - u)
    fits = recon.coefficients.fits
    optimal = _centered_optimal(fits.optimal, v)
    l1 = fits.left[2][1] * v[1] + fits.left[2][2] * v[2]
    l2 = fits.left[3][1] * v[1] + fits.left[3][2] * v[2]
    c1 = fits.center[2][3] * (v[4] - v[2])
    c2 = fits.center[3][1] * (v[2] + v[4])
    r1 = fits.right[2][2] * v[4] + fits.right[2][3] * v[5]
    r2 = fits.right[3][2] * v[4] + fits.right[3][3] * v[5]
    pl = (zero(l1), l1, l2, zero(l1), zero(l1))
    pc = (zero(c1), c1, c2, zero(c1), zero(c1))
    pr = (zero(r1), r1, r2, zero(r1), zero(r1))
    third = oftype(recon.epsilon, 1 // 3)
    p0 = ntuple(j -> 2 * optimal[j] - third * (pl[j] + pc[j] + pr[j]), Val(5))
    candidates = (p0, pl, pc, pr)

    half = one(recon.epsilon) / 2
    sixth = one(recon.epsilon) / 6
    d = (half, sixth, sixth, sixth)
    ω = _cweno_weights(recon, optimal, candidates, d)
    a1 = ω[1] * p0[2] + ω[2] * l1 + ω[3] * c1 + ω[4] * r1
    a2 = ω[1] * p0[3] + ω[2] * l2 + ω[3] * c2 + ω[4] * r2
    # Only P0 contributes cubic and quartic terms; the other candidates are
    # quadratic. Avoid multiplying and summing their known zero coefficients.
    a3, a4 = ω[1] * p0[4], ω[1] * p0[5]
    a = (zero(a1), a1, a2, a3, a4)

    # Enforce the central interpolation/conservation constraint to roundoff.
    a0 = _constant_coefficient(recon.input, u, a)
    return CellPolynomial((a0, a[2], a[3], a[4], a[5]))
end

@inline left(recon::CWENO5, um2::Real, um1::Real, u::Real, up1::Real, up2::Real) =
    value(cell_polynomial(recon, um2, um1, u, up1, up2), 1 // 2)

@inline left(recon::CWENO3, um::Real, u::Real, up::Real) =
    value(cell_polynomial(recon, um, u, up), 1 // 2)
