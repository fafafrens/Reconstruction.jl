# Complete-update benchmark

Measured on Apple M2, Julia 1.12.6, one thread, 4096 Float64 cells.
Median of 15 samples of 100 updates each, after warmup. All updates read the
same old state. Polynomial construction, flux evaluation and solution writes
are timed; reconstruction-object setup, scratch allocation, compilation and
periodic halo filling are excluded.

The numerical flux is scalar Burgers Rusanov, which uses both face states.
These timings do not predict a system solver's Riemann-flux cost. Point-input
rows compare computational work, not a validated cell-average discretization.

- Buffered: direct face calls, stored fluxes, then cell updates.
- Face roll: direct face calls, carry the preceding flux, write each cell once.
- Face add: direct face calls, add each face contribution to both neighbors.
- Poly roll: carry the preceding polynomial and flux; no flux array.
- Poly add: carry the polynomial, add each face contribution to both neighbors.
- Poly batch: construct each cell polynomial once, immediately store its two
  boundary states, then compute fluxes and update cells. No polynomial array;
  two state arrays and one flux array are reused.
- + center: Poly roll plus old-state center values and physical derivatives.
  This is a total time, not an incremental cost.

All supported methods are checked against independently indexed periodic
reference updates, with conservation, unchanged inputs, and center comparisons.
Wraparound checks also cover 1, 2 and 17 cells. Every timed sweep allocates zero
bytes; scratch arrays are preallocated. Face accumulation loops are serial.

In this run, batched polynomial evaluation was fastest for the slope limiters
and CWENO methods. The no-flux-array rolling polynomial loop was slower than
direct face reconstruction. Loop structure therefore matters as much as the
number of reconstructions; these timings alone do not establish the compiler's
exact cause. Repeat on the target machine and with the actual numerical flux.

```text
Source: /Users/eduardogrossi/finitevolume/Reconstruction.jl
Julia 1.12.6; apple-m2; threads=1
4096 periodic Float64 cells; microseconds per complete scalar Burgers update
Includes fluxes and output writes; excludes halo setup, construction and compilation.
Point-input rows compare work only, not a cell-average finite-volume discretization.
Profile Method              Input      Buffered  Face roll   Face add  Poly roll   Poly add Poly batch   + center
smooth  Godunov             averages       2.37       2.29       5.43       4.93       5.21       3.86       5.22
smooth  Minmod              averages       5.20       5.28      13.06       9.75       9.68       4.63      10.15
smooth  GeneralizedMinmod   averages      10.01      10.14      23.46      16.31      15.14       6.38      15.47
smooth  VanLeer             averages       6.30       6.32      13.33      10.40      10.08       5.00      11.10
smooth  VanAlbada           averages       6.95       7.25      15.63      11.26      10.79       5.16      11.93
smooth  MC                  averages       9.79       9.78      23.07      14.85      14.94       6.41      15.49
smooth  Superbee            averages       9.09       9.24      22.42      14.23      13.80       5.95      15.04
smooth  MP5                 averages      25.27      25.88      26.50          —          —          —          —
smooth  WENO3               points        10.83      10.74      21.69          —          —          —          —
smooth  WENO-Z              points        21.17      22.69      44.67          —          —          —          —
smooth  CWENO3              averages      26.77      27.28      56.28      40.82      39.65      15.24      42.96
smooth  CWENO5              averages      64.05      64.34     126.05      84.26      84.31      33.48      89.73
smooth  CWENO-Z3            averages      29.67      30.12      61.22      39.38      38.51      16.43      40.54
smooth  CWENO-Z5            averages      64.78      71.42     136.16      84.93      81.86      34.25      88.31
smooth  CWENO-Z3            points        27.49      28.50      60.29      40.08      39.21      16.59      40.61
smooth  CWENO-Z5            points        62.87      66.03     131.92      83.97      81.35      34.32      88.81
mixed   Godunov             averages       2.73       2.48       5.82       5.45       5.39       3.98       5.21
mixed   Minmod              averages       5.75       5.40      13.74       9.94      10.52       4.40      10.58
mixed   GeneralizedMinmod   averages      10.17      10.17      24.68      15.01      15.76       6.59      16.64
mixed   VanLeer             averages       6.68       6.70      13.80      10.84      10.37       5.17      11.61
mixed   VanAlbada           averages       7.10       7.51      15.59      11.88      10.85       5.56      12.28
mixed   MC                  averages      10.25       9.96      23.94      15.77      15.11       6.45      15.56
mixed   Superbee            averages       9.42       9.57      23.28      14.45      13.89       6.05      15.21
mixed   MP5                 averages      25.23      25.94      26.55          —          —          —          —
mixed   WENO3               points        11.07      10.64      21.49          —          —          —          —
mixed   WENO-Z              points        21.20      22.30      44.08          —          —          —          —
mixed   CWENO3              averages      26.77      27.25      55.03      41.06      39.87      15.13      43.19
mixed   CWENO5              averages      61.52      64.26     125.62      84.27      78.61      31.55      87.76
mixed   CWENO-Z3            averages      29.26      29.90      60.06      39.44      38.67      16.36      40.79
mixed   CWENO-Z5            averages      63.87      68.14     133.42      84.38      81.19      34.76      89.22
mixed   CWENO-Z3            points        27.74      28.62      56.96      37.80      36.54      15.82      39.76
mixed   CWENO-Z5            points        60.99      63.71     127.36      81.14      76.17      32.28      84.61
rough   Godunov             averages       2.38       2.36       5.41       4.87       5.20       3.70       5.12
rough   Minmod              averages      14.91       5.26      13.07      10.14       9.80       4.33      10.04
rough   GeneralizedMinmod   averages      10.03      10.25      22.92      14.80      15.08       6.79      15.40
rough   VanLeer             averages       6.30       6.35      13.34      10.34       9.93       5.06      11.35
rough   VanAlbada           averages       6.91       6.99      15.07      11.16      10.72       5.14      11.69
rough   MC                  averages      10.00       9.87      22.99      15.31      15.03       6.43      15.48
rough   Superbee            averages       9.29       9.21      22.46      14.45      13.80       6.04      15.06
rough   MP5                 averages      72.64      72.52      74.95          —          —          —          —
rough   WENO3               points        10.82      10.61      21.45          —          —          —          —
rough   WENO-Z              points        21.36      22.31      43.90          —          —          —          —
rough   CWENO3              averages      26.84      27.14      55.16      40.76      41.27      15.28      42.72
rough   CWENO5              averages      61.88      65.45     127.24      85.25      79.02      31.62      88.17
rough   CWENO-Z3            averages      29.73      30.05      60.79      39.74      39.13      16.73      40.80
rough   CWENO-Z5            averages      64.42      67.90     132.86      83.89      81.62      34.61      88.73
rough   CWENO-Z3            points        27.84      29.00      57.04      37.67      36.63      15.41      39.28
rough   CWENO-Z5            points        61.23      63.84     127.17      81.40      75.87      32.21      84.51
All measured updates allocate zero bytes. + center includes Poly roll and old-state center values/gradients.
```
