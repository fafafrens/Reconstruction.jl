# Reusable cell polynomials and lower-order reconstruction methods.

"""
    CellPolynomial(coefficients)

Polynomial in the local coordinate `ξ = (x - x_i)/dx`. Coefficients are an
immutable tuple in ascending order: `(a₀, a₁, ...)`. The cell occupies
`-1/2 ≤ ξ ≤ 1/2`. Use `value`, `derivative`, and `center` to evaluate it.
"""
struct CellPolynomial{N,T<:Number}
    coefficients::NTuple{N,T}
    function CellPolynomial(coefficients::NTuple{N,T}) where {N,T<:Number}
        N > 0 || throw(ArgumentError("a polynomial needs at least one coefficient"))
        return new{N,T}(coefficients)
    end
end

CellPolynomial(coefficients::Tuple{Vararg{Number}}) =
    CellPolynomial(promote(coefficients...))
CellPolynomial(::Tuple{}) = throw(ArgumentError("a polynomial needs at least one coefficient"))

"""Evaluate a `CellPolynomial` at the dimensionless local coordinate `ξ`."""
@inline value(p::CellPolynomial, ξ::Number) = evalpoly(ξ, p.coefficients)

"""
    derivative(p::CellPolynomial, ξ; dx)

Physical first derivative `dp/dx` at local coordinate `ξ`. The cell width `dx`
must be finite and positive. The reconstruction weights are held fixed when
differentiating the cell polynomial.
"""
@inline function derivative(p::CellPolynomial{N}, ξ::Number; dx::Real) where {N}
    _check_dx(dx)
    N == 1 && return zero(p.coefficients[1]) / dx
    coefficients = ntuple(j -> j * p.coefficients[j + 1], Val(N - 1))
    return evalpoly(ξ, coefficients) / dx
end

"""
    center(p::CellPolynomial; dx)
    center(recon, stencil...; dx)

Return `(value=uc, derivative=dudx)` at the cell center. Supported reconstruction
methods are Godunov (one sample), the slope limiters and CWENO3 (three samples),
and CWENO5 (five samples). WENO3 and WENOZ also implement `center` directly.
`dx` is the physical cell width.
"""
@inline center(p::CellPolynomial; dx::Real) =
    (value=p.coefficients[1], derivative=derivative(p, 0; dx))

"""
    face(p_i::CellPolynomial, p_ip1::CellPolynomial)

Return the left and right states at the shared face of two neighboring cells.
Each polynomial uses its own cell-centered local coordinate.
"""
@inline face(p::CellPolynomial, q::CellPolynomial) =
    (value(p, 1 // 2), value(q, -1 // 2))

"""
    cell_polynomial(recon, stencil...)
    cell_polynomial(recon, stencil::Tuple)
    cell_polynomial(recon, stencil::AbstractVector)

Build a polynomial centered on the middle stencil sample. Godunov needs one
sample, slope limiters and CWENO3 three, and CWENO5 five. The polynomial can be reused for
both cell boundaries, interior values, and physical derivatives.

Godunov and slope limiters preserve the central sample both as a point value
and as a cell average. CWENO3 and CWENO5 use their explicit `input` convention. Existing
WENO3 and WENOZ provide faces and centers without a cell polynomial; MP5 provides faces only.
"""
@inline cell_polynomial(::Godunov, u::Number) = CellPolynomial((u,))

@inline function cell_polynomial(lim::AbstractSlopeLimiter, um::Number, u::Number, up::Number)
    return CellPolynomial(promote(u, slope(lim, um, u, up)))
end
