# Keep the original acceptance test and limiter as a regression reference for
# changes to the fast path. Exercise both accepted and limited reconstructions.
function reference_minmod4(a, b, c, d)
    if (a > 0) & (b > 0) & (c > 0) & (d > 0)
        return min(a, b, c, d)
    elseif (a < 0) & (b < 0) & (c < 0) & (d < 0)
        return max(a, b, c, d)
    end
    return zero(promote_type(typeof(a), typeof(b), typeof(c), typeof(d)))
end

function reference_mp5(lim, um2, um1, u, up1, up2)
    uf = (2um2 - 13um1 + 47u + 27up1 - 3up2) / 60
    ump = u + minmod(up1 - u, lim.alpha * (u - um1))
    norm = sqrt(um2^2 + um1^2 + u^2 + up1^2 + up2^2)
    (uf - u) * (uf - ump) <= lim.tolerance * norm && return uf

    dm, d0, dp = um2 - 2um1 + u, um1 - 2u + up1, u - 2up1 + up2
    dmp = reference_minmod4(4d0 - dp, 4dp - d0, d0, dp)
    dmm = reference_minmod4(4d0 - dm, 4dm - d0, d0, dm)
    uul = u + lim.alpha * (u - um1)
    umd = (u + up1) / 2 - dmp / 2
    ulc = u + (u - um1) / 2 + 4 // 3 * dmm
    lo = max(min(u, up1, umd), min(u, uul, ulc))
    hi = min(max(u, up1, umd), max(u, uul, ulc))
    return uf + minmod(lo - uf, hi - uf)
end

mp5_face_allocations(lim, u) = @allocated face(lim, u...)

@testset "MP5 fast path and limiter regression" begin
    rng = MersenneTwister(73)
    for args in Iterators.product(ntuple(_ -> (-Inf, -2.0, -0.0, 0.0, 2.0, Inf, NaN), 4)...)
        @test isequal(minmod(args...), reference_minmod4(args...))
    end
    for a in (-Inf, -2.0, -0.0, 0.0, 2.0, Inf, NaN), b in (-Inf, -2.0, -0.0, 0.0, 2.0, Inf, NaN)
        @test Reconstruction._mp5_curvature(a, b) == reference_minmod4(4a - b, 4b - a, a, b)
    end
    profiles = (
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        (4.0, 1.0, 0.0, 1.0, 4.0, 9.0),
        ntuple(_ -> randn(rng), 6),
    )
    random_profiles = [ntuple(_ -> randn(rng), 6) for _ in 1:100]
    for T in (Float32, Float64), alpha in (1, 4, 7), tolerance in (0, 1e-12, 1e-4, -1e-4)
        lim = MP5(; alpha=T(alpha), tolerance=T(tolerance))
        for values in (profiles..., random_profiles...)
            u = map(T, values)
            expected = (reference_mp5(lim, u[1:5]...), reference_mp5(lim, reverse(u[2:6])...))
            actual = face(lim, u...)
            @test collect(actual) ≈ collect(expected) rtol=128eps(T) atol=128eps(T)
        end
        if T === Float64
            # Existing MP5 arithmetic mixes Float32/Float64 between branches;
            # the stable Float64 path is the allocation-free benchmark target.
            stencil = map(T, profiles[3])
            mp5_face_allocations(lim, stencil)
            @test mp5_face_allocations(lim, stencil) == 0
        end
    end
end
