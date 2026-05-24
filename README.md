# Reconstruction.jl

[![CI](https://github.com/fafafrens/Reconstruction.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/fafafrens/Reconstruction.jl/actions/workflows/ci.yml)

Finite-volume reconstruction and slope-limiter routines for face states.

The package uses small reconstruction objects instead of a large family of
function names. Each object carries its stencil metadata, so a solver can ask
how much halo data it needs before reconstructing faces.

## Reconstructions

- `Godunov()` / `PiecewiseConstant()`
- `WENO3()`
- `WENOZ()`
- `MP5(; alpha=4.0, tolerance=0.0)`
- `MinmodLimiter()`
- `GeneralizedMinmodLimiter(; theta=1.5)`
- `VanLeerLimiter()`
- `VanAlbadaLimiter()`
- `MonotonizedCentralLimiter()`
- `SuperbeeLimiter()`

## Metadata

```julia
using Reconstruction

recon = WENOZ()

halo_width(recon)        # 3
stencil_size(recon)      # 6
left_stencil_size(recon) # 5
```

The current convention is:

- `halo_width(recon)` is the number of ghost cells needed on each side.
- `stencil_size(recon)` is the number of values consumed by `face`.
- `left_stencil_size(recon)` is the number of values consumed by `left`.

## Scalar Face Reconstruction

`face(recon, ...)` returns `(left_state, right_state)` at the `i + 1/2` face.

```julia
left_state, right_state = face(WENO3(), u_im1, u_i, u_ip1, u_ip2)

left_state, right_state = face(WENOZ(),u_im2,u_im1,u_i,u_ip1,u_ip2,u_ip3)
```

Tuple and vector stencils are also accepted:

```julia
stencil = (u_im1, u_i, u_ip1, u_ip2)
left_state, right_state = face(WENO3(), stencil)
```

## Array Face Reconstruction

The same API works on arrays with matching axes:

```julia
left, right = face(VanLeerLimiter(), u_im1, u_i, u_ip1, u_ip2)
```

For allocation-free use, pass output arrays:

```julia
face!(VanLeerLimiter(), left, right, u_im1, u_i, u_ip1, u_ip2)
```

`reconstruct` and `reconstruct!` are aliases for `face` and `face!`.

## Left States, Slopes, And Increments

```julia
left(WENOZ(), u_im2, u_im1, u_i, u_ip1, u_ip2)
slope(VanLeerLimiter(), u_im1, u_i, u_ip1)
increment(MP5(), u_im2, u_im1, u_i, u_ip1, u_ip2)
```

## Running Tests

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Demo

```sh
julia --project=. examples/reconstruction_demo.jl
```

The demo prints reconstruction-error and timing tables. If `Plots.jl` is
available in the active environment, it also writes diagnostic plots under
`examples/reconstruction_plots/`.
