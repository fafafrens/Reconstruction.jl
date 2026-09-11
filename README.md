# Reconstruction.jl

[![CI](https://github.com/fafafrens/Reconstruction.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/fafafrens/Reconstruction.jl/actions/workflows/ci.yml)

Uniform-grid reconstruction in Julia. Load the module with:

```julia
using Reconstruction
```

The module requires `MuladdMacro` and the `LinearAlgebra` standard library.

## Face reconstruction and input conventions

`face(recon, stencil...)` returns the left and right states at a shared face.
The existing `left`, `right`, `face!`, `reconstruct`, and `reconstruct!` APIs remain
available. The stencil is ordered from left to right.

| Reconstruction | Input convention | Face stencil / halo width |
| --- | --- | --- |
| Godunov | Point values or cell averages | 2 / 1 |
| Slope limiters | Point values or cell averages | 4 / 2 |
| WENO3, WENOZ | Point values | 4 / 2, 6 / 3 |
| MP5 | Cell averages for state reconstruction | 6 / 3 |
| CWENO3 | Explicit `PointValues()` or `CellAverages()` | 4 / 2 |
| CWENO5 | Explicit `PointValues()` or `CellAverages()` | 6 / 3 |

The MP5 formula can also reconstruct numerical fluxes in a finite-difference
scheme; that interpretation does not make it ordinary interpolation of point
values. WENOZ's corrected optimal point-interpolation weights are
`(1/16, 10/16, 5/16)`.

## Reusable cell polynomials

Godunov, the slope limiters, CWENO3, and CWENO5 can return a `CellPolynomial`. Its
coefficients are an immutable tuple in ascending powers of the dimensionless
coordinate `ξ = (x - x_i)/dx`, with cell boundaries at `ξ = ±1/2`.

```julia
# Construct once, outside the loop. The default input is CellAverages().
recon = CWENO5(; input=CellAverages(), epsilon=1e-6)

# Five averages, centered on cell i. Use your simulation data here.
u = (0.2, 0.4, 0.5, 0.8, 0.9)
p = cell_polynomial(recon, u)  # Also accepts splatted scalars or a vector.

uc, dudx = center(p; dx=0.1)
at_right_boundary = value(p, 0.5)
at_interior_point = value(p, 0.2)
physical_derivative = derivative(p, 0.2; dx=0.1)

# Convenience wrapper when only center quantities are needed:
uc, dudx = center(recon, u; dx=0.1)
```

`center` returns the named tuple `(value=uc, derivative=dudx)`. Physical
derivatives require a finite positive `dx`. With cell-average input, `uc` is a
reconstructed point value, which generally differs from the stored average.
The polynomial's integral over the local cell preserves that average. With
point-value input, the center value equals the central sample.

For a shared face, evaluate the right boundary of cell `i` and the left
boundary of cell `i+1`:

```julia
q = cell_polynomial(recon, 0.4, 0.5, 0.8, 0.9, 1.1)
ul, ur = face(p, q)

# Equivalent six-sample face interface:
ul, ur = face(recon, 0.2, 0.4, 0.5, 0.8, 0.9, 1.1)
```

Reuse `p` for multiple evaluations to avoid recomputing the nonlinear weights.
Building all physical face states still requires three ghost cells per side;
building only CWENO5 polynomials for the physical cells requires two.

For a smaller stencil, CWENO3 builds a quadratic from three samples:

```julia
recon3 = CWENO3(; input=CellAverages(), epsilon=1e-6)
p3 = cell_polynomial(recon3, 0.4, 0.5, 0.8)
uc, dudx = center(p3; dx=0.1)
ul, ur = face(recon3, 0.4, 0.5, 0.8, 0.9)
```

CWENO3 requires two ghost cells per side for all physical face states, or one
for only the physical cells' polynomials. The same tuple, vector, `left`,
`right`, and array face interfaces are available for both CWENO orders.

The same interface works for lower-order reconstructions:

```julia
p0 = cell_polynomial(Godunov(), 1.0)                 # Constant
p1 = cell_polynomial(MinmodLimiter(), 0.8, 1.0, 1.3) # Linear
uc, dudx = center(p1; dx=0.1)
```

The linear coefficient is the existing limited `slope`, an increment across
one cell. Dividing it by `dx` gives the physical derivative. WENO3, WENOZ, and
MP5 retain their face interfaces and expose no cell polynomial. WENO3 and WENOZ
do provide a `center` reconstruction, described below; MP5 provides neither.

## WENO centre gradients

WENO3 and WENOZ accept `center(recon, stencil...; dx)`, returning the same
`(value=uc, derivative=dudx)` named tuple as the polynomial reconstructions.
The stencil is the centred one, three samples for WENO3 and five for WENOZ.

```julia
uc, dudx = center(WENOZ(), 0.2, 0.4, 0.5, 0.8, 0.9; dx=0.1)
uc, dudx = center(WENO3(), 0.4, 0.5, 0.8; dx=0.1)
```

`uc` is exactly the central sample. Every WENO substencil contains node `i`, so
each candidate interpolates `u` at `ξ = 0` and any convex blend returns it
whatever the nonlinear weights do. All of the reconstruction content is in the
derivative.

The candidates are the derivatives at `ξ = 0` of the same substencil polynomials
the face reconstruction blends, and the smoothness indicators and `τ` are shared
with it. Only the linear weights differ, because the face weights are optimal at
`ξ = 1/2` rather than at the centre:

| Reconstruction | Face weights | Centre-derivative weights | Derivative order |
| --- | --- | --- | --- |
| WENO3 | `(1/4, 3/4)` | `(1/2, 1/2)` | 2 |
| WENOZ | `(1/16, 10/16, 5/16)` | `(1/6, 2/3, 1/6)` | 4 |

Reusing the face weights at the centre drops WENOZ's derivative from fourth
order to second, so the two sets are not interchangeable. All weights are
positive, so no negative-weight splitting is needed. The tests derive them in
exact rational arithmetic against the corresponding central differences.

Orders are measured at generic smooth points. Near a critical point the
indicators collapse toward zero, `τ/β` stops being small, and the nonlinear
weights drift off the linear ones; the WENOZ derivative then converges
erratically from step to step while still falling monotonically. WENO-Z makes
the same trade at faces. At a discontinuity the blend leans toward the smoother
substencil, and the result is a derivative of the reconstruction rather than a
classical derivative of the data.

Neither reconstruction exposes a `cell_polynomial`, and this is deliberate. The
face and centre operators use different linear weights, so no single polynomial
reproduces both. Evaluating such a polynomial at `ξ = 1/2` would disagree with
`left`. Use `CWENO3` or `CWENO5` when a genuine per-cell polynomial is needed,
which is the problem CWENO exists to solve. MP5 gets no `center` at all: its
limiter clips a face value against bounds built from the neighbouring cell
across that face, and has no meaning away from it.

## CWENO construction and accuracy

CWENO5 follows [Cravero et al., section 3](https://arxiv.org/abs/1607.07319):
a centered quartic, three quadratic substencils, and an additional quartic
candidate reproduce the optimal polynomial under the linear weights
`(1/2, 1/6, 1/6, 1/6)`. The default `JSWeights()` uses exponent two and the smoothness
indicator of each candidate, including the additional quartic. This is ordinary
CWENO, not CWENO-Z.

CWENO3 uses an optimal quadratic and two linear substencils, plus an additional
quadratic candidate, with linear weights `(1/2, 1/4, 1/4)`. It uses the same
Jiang–Shu weighting rule as CWENO5. The polynomial degrees and smooth accuracy
targets are:

| Method | Polynomial degree | Value order | First derivative order |
| --- | --- | --- | --- |
| CWENO3 | 2 | 3 | 2 |
| CWENO5 | 4 | 5 | 4 |

Fitting matrices and smoothness integrals are derived with exact rational
arithmetic when each reconstruction object is constructed, then converted and
stored as immutable tables in that object. Only the requested order and input
convention are built; the additional τ table is built only for `ZWeights()`.
There are no global coefficient caches. Construct once outside the cell loop
and reuse the object, since constructing another object repeats this setup.
Reconstruction uses fixed tuple operations, with no linear
solves, quadrature, or heap allocations for the tested Float32/Float64 scalar
paths after compilation. Use `epsilon=1f-6` to select Float32 tables; input types
otherwise follow Julia's numeric promotion rules.

The default `epsilon=1e-6` is intended for data of order one.
It has units of the squared reconstructed variable, so choose it consistently
with data scaling. For nondimensional problems, a mesh-dependent choice such as
`epsilon=dx^2` is also available by constructing the object with that value.
Convergence tests cover this choice and smooth critical points. Gradients at
discontinuities are derivatives of the reconstructed polynomial, not a classical
derivative of the discontinuous function. CWENO does not guarantee positivity
or TVD bounds. Nonuniform grids would require geometry-dependent tables.

## CWENO-Z weights

Both CWENO orders support a weighting strategy stored in the reconstruction's
concrete type. Existing constructors continue to default to `JSWeights()`.

```julia
recon3z = CWENO3(; weights=ZWeights(), input=CellAverages())
recon5z = CWENO5(; weights=ZWeights(), input=PointValues())
p = cell_polynomial(recon5z, 0.2, 0.4, 0.5, 0.8, 0.9)
uc, dudx = center(p; dx=0.1)
```

`ZWeights(; power=2)` uses `αₖ = dₖ * (1 + (τ/(βₖ+epsilon))^power)`, followed
by normalization. Powers one and two are supported; two is the default for both
orders. Stencils, polynomial degrees, conservation/interpolation constraints,
and value/derivative accuracy targets are the same as for ordinary CWENO.
The face and polynomial APIs are shared, including the generic mirrored `right`.

The central indicator is `βopt = I[Popt]`, while the blended polynomial still
uses the additional candidate `P₀`. This differs from our default JS weights,
which measure `I[P₀]` for that candidate. With `βL`, `βC`, and `βR` denoting the
lower-degree substencil indicators, the global indicators are:

| Reconstruction | Global smoothness indicator |
| --- | --- |
| CWENO-Z3, either input | `abs(βL + βR - 2βopt)` |
| CWENO-Z5, cell averages | `abs(βL + 4βC + βR - 6βopt)` |
| CWENO-Z5, point values | `abs(βL + 6βC + βR - 8βopt)` |

The cell-average formulas follow equations (19) and (20) of
[Cravero, Semplice, and Visconti](https://www.igpm.rwth-aachen.de/Download/reports/pdf/IGPM484.pdf).
The point-value fifth-order formula is derived here for our interpolation
polynomials: using the cell-average combination would leave an order-four term.
The corrected combination cancels that term and gives `τ = O(dx^6)` in smooth
regions; CWENO-Z3 has `τ = O(dx^4)` for both conventions.

These cancellations are derived with exact rational arithmetic. Each Z
reconstruction stores an additional immutable table that evaluates τ directly
from the optimal polynomial's coefficients, avoiding subtraction of nearly equal
smoothness indicators at runtime. Tests verify the defining combinations and
their cancellation orders exactly. Normalization uses bounded ratios so that a
very small positive epsilon does not overflow the weight formula.

The same epsilon scaling considerations apply to both weighting strategies.
Tests cover powers one and two, both input conventions, smooth critical points,
discontinuities, and zero-allocation Float32/Float64 reconstruction. CWENO-Z can
reduce errors compared with ordinary CWENO, but it is not uniformly better for
every profile or parameter choice and does not guarantee TVD or positivity.

## Performance

The CWENO kernels exploit symmetry and structural zeros in the fitting and
smoothness tables. Z-weight powers one and two use explicit scalar operations,
and normalization uses one final reciprocal. The kernels use fused multiply-add
operations, so results can differ from the dense formulas by floating-point
roundoff. Tables are still constructed once per reconstruction object.

Run the reproducible face-sweep benchmark with:

```sh
julia --project=. benchmarks_cweno.jl
# Optional: use the same harness with an earlier source directory.
julia --project=. benchmarks_cweno.jl /path/to/earlier/source
```

For 4096 periodic Float64 point samples on Apple aarch64 with Julia 1.12.6,
median times for the smooth profile were:

| Method | Relative cost |
| --- | ---: |
| WENOZ, face sweep | 1.0 |
| CWENO5, ZWeights(power=2), face sweep | 3.0 |
| CWENO5, ZWeights(power=2), cached polynomial sweep | 3.2 |

Costs are given relative to the WENOZ face sweep in the same process, because
absolute times on a laptop move by a factor of two with thermal and turbo state.
One run measured 18 µs, 55 µs, and 57 µs for the three rows. The cached sweep
includes both polynomial construction and face evaluation. Separate coefficient
arrays cost about 1.1 times the cached sweep. All measured sweeps allocate zero
bytes after compilation; construction and output allocation are excluded. A
profile containing jumps gives similar timings. These are local measurements,
not a guarantee for other machines or input types. WENOZ and CWENO-Z use
different reconstruction formulas.

`left` and `center` for WENO3 and WENOZ carry `@inline`. Without it Julia emits
them as out-of-line calls, which blocks vectorization of the surrounding sweep
and costs a factor of about 2.7 on both face and centre sweeps. MP5 now inlines
only its small acceptance path and keeps the full limiter out of line; inlining
the entire limiter measured slower on smooth data. Inlining changes how the
fused multiply-adds contract, so face values
move by up to about three machine epsilon relative to the stencil magnitude.

Reuse a local polynomial when requesting faces, centers, derivatives, or
multiple interior values from the same cell. Storing every polynomial in an
array is not necessarily faster for a face-only sweep: compiler vectorization
and memory traffic also matter. The benchmark compares direct faces, an array
of polynomials, and separate coefficient arrays.

For comparisons that include MP5 and both CWENO input conventions, run:

```sh
julia --project=. benchmarks_faces.jl
julia --project=. benchmarks_faces.jl /path/to/earlier/source
```

This benchmark uses identical source-loading paths, constructs reconstruction
objects outside the loops, and measures smooth, mixed, and random rough data.
MP5 receives cell averages; WENO-Z receives point values. The reported face
sweep produces both states at every shared face.

CWENO-Z normalization uses `min(τ, m)` in place of `τ*m/max(τ, m)`, where
`m = minimum(β + epsilon)`. This removes one division and the per-candidate
products while retaining bounded ratios for extreme data scales. MP5 accepts
already admissible candidates before computing a norm, omits the norm entirely
when tolerance is zero, and keeps the full limiter in a separate function.
Its internal curvature limiter uses extrema to detect a common sign.

These optimizations reduce CWENO-Z normalization work and accelerate MP5's
smooth and limited paths. Compare both versions using this benchmark: older
measurements taken before WENO's explicit inlining substantially overstate its
cost, so their near-parity with CWENO-Z does not apply to the current methods.
These compare runtime with each method's existing input convention and defaults,
not identical error levels.

## Validation and plots

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. examples/test_and_plot_reconstruction_limiters.jl
```

The main suite includes fitting constraints, conservation, point interpolation,
reflection symmetry, stencil interfaces, discontinuities, convergence, numeric
types, and allocation checks. The WENO centre gradients add exact rational
derivations of their linear weights, convergence at generic and near-critical
points, and Float32 and zero-allocation checks. An additional 480 randomized comparisons check
the optimized kernels against the dense matrix definition for both orders,
input conventions, weight strategies, and Float32/Float64. Plotting additionally
needs `Plots`; when it is
unavailable, the plotting script still runs its smoke tests and skips figures.
Additional regression checks compare MP5 with its original limiter on smooth,
discontinuous, and random stencils, including nonzero tolerance and Float32/64.
CWENO normalization is also compared with a BigFloat reference across extreme
scales. Allocation-free MP5 sweeps are checked for Float64; its existing mixed
Float32/Float64 branch arithmetic remains unchanged.

Diagnostics use exact cell averages for Godunov, slope limiters, MP5, and
cell-average CWENO3/CWENO5, and point samples for WENO3, WENOZ, and point-value
CWENO3/CWENO5.
They include face states and jumps for several profiles, plus center values and
physical derivatives for a smooth sine wave.

The original face API examples and timing demo remain available via
`julia --project=. examples/reconstruction_demo.jl`.
