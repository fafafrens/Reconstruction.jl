"""
    PPM()

Classic monotone piecewise-parabolic reconstruction from cell averages on a
uniform grid. Uses limited centered slopes for the fourth-order interface
estimates, then limits each cell's parabola while preserving its average.

`cell_polynomial` uses five samples `(u_im2, u_im1, u_i, u_ip1, u_ip2)`;
`face` uses six samples and requires three ghost cells per side. The polynomial
is third-order accurate at general interior points in smooth monotone regions;
classic limiting flattens extrema, including smooth ones.

This is spatial reconstruction only: no characteristic tracing, shock
flattening, or contact steepening is included. See Sekora and Colella (2009),
sections 1.2 and 1.4, https://arxiv.org/abs/0903.4200.
"""
struct PPM <: AbstractReconstruction{3,6} end

@inline _ppm_slope(a, b) = minmod((a + b) * (1 // 2), 2a, 2b)

@muladd @inline function cell_polynomial(::PPM, um2::Real, um1::Real, u::Real, up1::Real, up2::Real)
    dm2, dm, dp, dp2 = um1 - um2, u - um1, up1 - u, up2 - up1
    sm, s, sp = _ppm_slope(dm2, dm), _ppm_slope(dm, dp), _ppm_slope(dp, dp2)

    # Boundary offsets relative to the central average, equations (7)-(8).
    a = -(1 // 2) * dm - (1 // 6) * (s - sm)
    b = (1 // 2) * dp - (1 // 6) * (sp - s)

    # Equation (27), using sign and magnitude tests without squared products.
    if !((a < 0 < b) || (b < 0 < a))
        a, b = zero(a), zero(b)
    elseif abs(a) > 2abs(b)
        a = -2b
    elseif abs(b) > 2abs(a)
        b = -2a
    end

    # Integral over [-1/2,1/2] is a0 + a2/12 = u.
    curvature = a + b
    return CellPolynomial(promote(u - (1 // 4) * curvature, b - a, 3curvature))
end

@inline left(recon::PPM, um2::Real, um1::Real, u::Real, up1::Real, up2::Real) =
    value(cell_polynomial(recon, um2, um1, u, up1, up2), 1 // 2)
