# Included by runtests_reconstruction_limiters.jl.

# Measured through a top-level wrapper, as the CWENO allocation tests do. An
# inline @allocated over a splat inside a testset loop boxes on Julia 1.11 and
# reports phantom allocations.
center_allocations(recon, u, dx) = @allocated center(recon, u...; dx)

@testset "WENO-Z centre gradient" begin
    # --- the linear weights, derived exactly rather than trusted --------------
    # Each substencil quadratic contributes its dp/dξ at ξ = 0 as an exact
    # rational row over the five samples (u_{i-2} … u_{i+2}).
    gL = (1 // 2, -2 // 1, 3 // 2, 0 // 1, 0 // 1)
    gC = (0 // 1, -1 // 2, 0 // 1, 1 // 2, 0 // 1)
    gR = (0 // 1, 0 // 1, -3 // 2, 2 // 1, -1 // 2)
    # The fourth-order central difference these must reproduce.
    target = (1 // 12, -2 // 3, 0 // 1, 2 // 3, -1 // 12)

    d = (1 // 6, 2 // 3, 1 // 6)
    @test sum(d) == 1
    @test all(w > 0 for w in d)                       # no negative-weight splitting
    blended = ntuple(k -> d[1] * gL[k] + d[2] * gC[k] + d[3] * gR[k], 5)
    @test blended == target                           # exact, not approximate

    # The face weights solve a different problem and must not reproduce it.
    face_d = (1 // 16, 10 // 16, 5 // 16)
    @test sum(face_d) == 1
    @test ntuple(k -> face_d[1] * gL[k] + face_d[2] * gC[k] + face_d[3] * gR[k], 5) != target

    # --- exactness on polynomials the method must reproduce -------------------
    for dx in (0.25, 1.0, 2.5)
        u = ntuple(j -> 3.5, 5)
        c = center(WENOZ(), u...; dx)
        @test c.value == 3.5
        @test c.derivative == 0.0                     # constants are exact

        slope0 = -1.75
        v = ntuple(j -> 2.0 + slope0 * (j - 3) * dx, 5)
        @test center(WENOZ(), v...; dx).derivative ≈ slope0 rtol = 1.0e-13
    end

    # --- the centre value is the central sample, whatever the data does -------
    for u in ((0.0, 0.0, 0.0, 9.0, 9.0), (1.0, -2.0, 0.75, 4.0, -3.0))
        @test center(WENOZ(), u...; dx=0.1).value == u[3]
    end

    # --- fourth-order convergence at generic smooth points --------------------
    f(x) = sin(x) + 0.3cos(3x)
    fp(x) = cos(x) - 0.9sin(3x)
    steps = (0.1, 0.05, 0.025, 0.0125)
    for (g, gp, x0) in ((f, fp, 0.0), (sin, cos, 1.0))
        errors = map(h -> abs(center(WENOZ(), ntuple(j -> g(x0 + (j - 3) * h), 5)...;
                                     dx=h).derivative - gp(x0)), steps)
        orders = ntuple(k -> log2(errors[k] / errors[k + 1]), length(steps) - 1)
        @test all(o -> 3.8 < o < 4.2, orders)
        @test last(errors) < 1.0e-8
    end

    # Near a critical point the indicators collapse toward zero, so τ₅/β is no
    # longer small and the nonlinear weights drift off the linear ones. WENO-Z
    # makes the same trade at faces. The per-step rate becomes erratic, so pin
    # down what actually holds: the error still falls monotonically, and the
    # rate averaged over the whole refinement stays above fourth order.
    x_crit = 0.7                                      # f'(0.7) ≈ -0.012
    fine = (0.1, 0.05, 0.025, 0.0125, 0.00625)
    e_crit = map(h -> abs(center(WENOZ(), ntuple(j -> f(x_crit + (j - 3) * h), 5)...;
                                 dx=h).derivative - fp(x_crit)), fine)
    @test all(e_crit[k + 1] < e_crit[k] for k in 1:length(fine) - 1)
    @test log2(first(e_crit) / last(e_crit)) / (length(fine) - 1) > 3.5

    # --- shocks stay finite and bounded by the one-sided slopes ---------------
    step = (0.0, 0.0, 0.0, 1.0, 1.0)
    c = center(WENOZ(), step...; dx=0.1)
    @test isfinite(c.derivative)
    @test 0.0 <= c.derivative <= 10.0                 # within the data's slope range

    # --- interface parity, argument validation, types, allocations ------------
    u = (0.2, 0.4, 0.5, 0.8, 0.9)
    @test center(WENOZ(), u; dx=0.1) == center(WENOZ(), u...; dx=0.1)
    @test center(WENOZ(), collect(u); dx=0.1) == center(WENOZ(), u...; dx=0.1)
    @test_throws DimensionMismatch center(WENOZ(), [1.0, 2.0]; dx=0.1)
    for bad in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError center(WENOZ(), u...; dx=bad)
    end

    u32 = map(Float32, u)
    c32 = @inferred center(WENOZ(), u32...; dx=0.1f0)
    @test c32.value isa Float32
    @test c32.derivative isa Float32                  # rationals keep Float32

    for (v, w) in ((u, 0.1), (u32, 0.1f0))
        center_allocations(WENOZ(), v, w)                 # Warm the measurement wrapper.
        @test center_allocations(WENOZ(), v, w) == 0
    end

    # Mirroring the stencil must flip the sign of the gradient.
    rev = reverse(u)
    @test center(WENOZ(), rev...; dx=0.1).derivative ≈
        -center(WENOZ(), u...; dx=0.1).derivative rtol = 1.0e-14
end

@testset "WENO3 centre gradient" begin
    # --- the linear weights, derived exactly rather than trusted --------------
    # The candidate lines through nodes (-1,0) and (0,1) contribute their
    # dp/dξ at ξ = 0, which are the one-sided increments.
    gL = (-1 // 1, 1 // 1, 0 // 1)
    gR = (0 // 1, -1 // 1, 1 // 1)
    target = (-1 // 2, 0 // 1, 1 // 2)     # second-order central difference

    d = (1 // 2, 1 // 2)
    @test sum(d) == 1
    @test all(w > 0 for w in d)
    @test ntuple(k -> d[1] * gL[k] + d[2] * gR[k], 3) == target

    face_d = (1 // 4, 3 // 4)              # optimal at ξ = 1/2, not at ξ = 0
    @test sum(face_d) == 1
    @test ntuple(k -> face_d[1] * gL[k] + face_d[2] * gR[k], 3) != target

    # --- exactness on polynomials the method must reproduce -------------------
    for dx in (0.25, 1.0, 2.5)
        @test center(WENO3(), 3.5, 3.5, 3.5; dx).value == 3.5
        @test center(WENO3(), 3.5, 3.5, 3.5; dx).derivative == 0.0

        slope0 = -1.75
        v = ntuple(j -> 2.0 + slope0 * (j - 2) * dx, 3)
        @test center(WENO3(), v...; dx).derivative ≈ slope0 rtol = 1.0e-13
    end

    for u in ((0.0, 0.0, 9.0), (1.0, -2.0, 0.75))
        @test center(WENO3(), u...; dx=0.1).value == u[2]
    end

    # --- second-order convergence at generic smooth points --------------------
    f(x) = sin(x) + 0.3cos(3x)
    fp(x) = cos(x) - 0.9sin(3x)
    # Coarse steps are still pre-asymptotic here, measuring about 2.34, so start
    # where the rate has settled.
    steps = (0.025, 0.0125, 0.00625, 0.003125)
    for (g, gp, x0) in ((f, fp, 0.0), (sin, cos, 1.0))
        stencils = map(h -> ntuple(j -> g(x0 + (j - 2) * h), 3), steps)
        errors = map((h, u) -> abs(center(WENO3(), u...; dx=h).derivative - gp(x0)),
                     steps, stencils)
        orders = ntuple(k -> log2(errors[k] / errors[k + 1]), length(steps) - 1)
        @test all(o -> 1.9 < o < 2.2, orders)

        # In smooth data the nonlinear weights must stay near (1/2, 1/2), so the
        # result should track the plain central difference rather than drift.
        for (h, u) in zip(steps, stencils)
            linear = (u[3] - u[1]) / (2h)
            @test center(WENO3(), u...; dx=h).derivative ≈ linear rtol = 1.0e-3
        end
    end

    # --- shocks stay finite and within the data's own slope range -------------
    c = center(WENO3(), 0.0, 0.0, 1.0; dx=0.1)
    @test isfinite(c.derivative)
    @test 0.0 <= c.derivative <= 10.0
    # The blend leans toward the flat side rather than the jump.
    @test c.derivative < 0.5 * (1.0 - 0.0) / 0.1

    # --- interface parity, argument validation, types, allocations ------------
    u = (0.4, 0.5, 0.8)
    @test center(WENO3(), u; dx=0.1) == center(WENO3(), u...; dx=0.1)
    @test center(WENO3(), collect(u); dx=0.1) == center(WENO3(), u...; dx=0.1)
    @test_throws DimensionMismatch center(WENO3(), [1.0, 2.0]; dx=0.1)
    for bad in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError center(WENO3(), u...; dx=bad)
    end

    u32 = map(Float32, u)
    c32 = @inferred center(WENO3(), u32...; dx=0.1f0)
    @test c32.value isa Float32
    @test c32.derivative isa Float32

    for (v, w) in ((u, 0.1), (u32, 0.1f0))
        center_allocations(WENO3(), v, w)                 # Warm the measurement wrapper.
        @test center_allocations(WENO3(), v, w) == 0
    end

    @test center(WENO3(), reverse(u)...; dx=0.1).derivative ≈
        -center(WENO3(), u...; dx=0.1).derivative rtol = 1.0e-14
end
