# PPM benchmark

Measured with Julia 1.12.6 on Apple M2, one thread, 4096 periodic Float64 cells.
Each entry is the median of 15 samples of 100 complete scalar Burgers updates.
Reconstruction (including polynomial construction), flux evaluation, and output
writes are timed. Reconstruction-object setup, scratch allocation, compilation,
and periodic halo filling are excluded. All rows below use cell averages.

The variants agree with an independent periodic reference and preserve the sum
of cell averages to roundoff. Checks also cover 1, 2, and 17-cell wraparound.
Every measured update allocates zero bytes. The center column includes both
the rolling polynomial update and old-state center values/derivatives.

Timings are from the same run; compare these rows with each other rather than
with older laptop measurements under different clock/thermal conditions.
Column definitions are in [updates_results.md](updates_results.md).

```text
Profile Method              Input      Buffered  Face roll   Face add  Poly roll   Poly add Poly batch   + center
smooth  MP5                 averages      43.19      44.40      46.04          —          —          —          —
smooth  PPM                 averages      60.87      63.01      89.81      53.40      53.00      30.57      55.10
smooth  CWENO3              averages      46.38      47.10      94.54      70.06      68.36      25.94      73.65
smooth  CWENO5              averages     104.62     110.02     216.36     144.32     133.97      53.90     150.14
smooth  CWENO-Z3            averages      50.19      51.15     103.53      67.76      66.14      27.91      70.14
smooth  CWENO-Z5            averages     110.33     115.88     228.18     144.56     139.95      59.01     152.66
mixed   PPM                 averages      59.83      61.88      89.58      53.16      52.45      30.28      54.85
rough   PPM                 averages      59.86      61.84      86.10      49.50      49.69      30.60      63.14
```

PPM here is the classic monotone spatial reconstruction, which clips smooth
extrema. Cost comparisons do not imply equal accuracy between reconstruction
methods or predict the cost of a system solver's numerical flux.
