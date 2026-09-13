module ExampleData
include(joinpath(@__DIR__, "..", "examples", "common.jl"))
end

@testset "example sampling and periodic reconstruction" begin
    N = 64
    name, f, primitive = first(ExampleData.PROFILES)
    x = ((0:N-1) .+ 0.5) ./ N
    points = f.(x)
    averages = sinc(1 / N) .* points
    @test ExampleData.sample_profile(WENOZ(), f, primitive, N) ≈ points
    @test all(isapprox.(ExampleData.sample_profile(MP5(), f, primitive, N), averages; atol=1e-14))
    @test all(isapprox.(ExampleData.sample_profile(CWENO5(), f, primitive, N), averages; atol=1e-14))
    @test ExampleData.sample_profile(CWENO5(input=PointValues()), f, primitive, N) ≈ points

    for (_, f, primitive) in ExampleData.PROFILES, recon in RECONS
        u = ExampleData.sample_profile(recon, f, primitive, N)
        l, r = ExampleData.reconstruct_periodic_faces(recon, u)
        @test length(l) == N
        @test length(r) == N
        @test all(isfinite, l)
        @test all(isfinite, r)
        if ExampleData.has_center(recon)
            c, d = ExampleData.reconstruct_periodic_centers(recon, u)
            @test length(c) == N
            @test length(d) == N
            @test all(isfinite, c)
            @test all(isfinite, d)
        end
    end
end
