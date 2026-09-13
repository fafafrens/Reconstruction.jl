# Run: julia --project=. benchmarks_faces.jl [earlier-source-directory]
# Use the same loading path for old/new sources to avoid import-context bias.
include("benchmarks/common.jl")

function run_one(label, recon, input, profile; N=4096)
    u = periodic_samples(recon, input, profile, N)
    l, r = zeros(N), zeros(N)
    us, bytes = measure_sweep(() -> direct_faces!(l, r, recon, u))
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
