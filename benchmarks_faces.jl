# Run: julia --project=. benchmarks_faces.jl [earlier-source-directory]
# Use the same loading path for old/new sources to avoid import-context bias.
const BENCHMARK_SOURCE = isempty(ARGS) ? (@__DIR__) : only(ARGS)
include(joinpath(BENCHMARK_SOURCE, "src", "Reconstruction.jl"))
using .Reconstruction
using Statistics, Printf, Random

@noinline function sweep!(l, r, recon, u)
    n = Val(stencil_size(recon))
    @inbounds for i in eachindex(l)
        l[i], r[i] = face(recon, ntuple(k -> u[i + k - 1], n)...)
    end
    return nothing
end

function measure(f; repetitions=100, samples=15)
    f()
    bytes = @allocated f()
    times = map(1:samples) do _
        (@elapsed for _ in 1:repetitions
            f()
        end) / repetitions
    end
    return 1e6median(times), bytes
end

function run_one(label, recon, input, profile; N=4096)
    radius = left_stencil_size(recon) ÷ 2
    dx = 1 / N
    rng = MersenneTwister(129)
    raw = [begin
        x = (i - 1) / N
        if profile == "rough"
            randn(rng)
        else
            a, b = input == "averages" ? (sinc(dx), sinc(3dx)) : (1.0, 1.0)
            a*sin(2π*x) + 0.2b*cos(6π*x) +
                (profile == "mixed" && 0.3 <= x < 0.7 ? 1.0 : 0.0)
        end
    end for i in 1:N]
    u = [raw[mod1(i - radius, N)] for i in 1:(N + 2radius + 1)]
    l, r = zeros(N), zeros(N)
    us, bytes = measure(() -> sweep!(l, r, recon, u))
    @assert bytes == 0
    @assert all(isfinite, l) && all(isfinite, r)
    @printf("%-6s %-18s %-8s %10.2f %6d\n", profile, label, input, us, bytes)
    flush(stdout)
end

function main()
    println("Source: ", BENCHMARK_SOURCE)
    println("Julia ", VERSION, "; ", Sys.CPU_NAME, "; threads=", Threads.nthreads())
    println("4096 cells; median microseconds per two-state face sweep; no setup/compilation")
    cases = (
        ("WENO-Z", WENOZ(), "points"),
        ("CWENO-Z5 p2", CWENO5(; input=PointValues(), weights=ZWeights()), "points"),
        ("CWENO-Z5 p2", CWENO5(; input=CellAverages(), weights=ZWeights()), "averages"),
        ("CWENO-Z3 p2", CWENO3(; input=PointValues(), weights=ZWeights()), "points"),
        ("MP5 tol=0", MP5(), "averages"),
        ("MP5 tol=1e-12", MP5(; tolerance=1e-12), "averages"),
    )
    for profile in ("smooth", "mixed", "rough"), (label, recon, input) in cases
        run_one(label, recon, input, profile)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
