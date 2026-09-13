# Accuracy tables and plots. Performance measurements live in benchmarks_*.jl.
using Printf
include("test_and_plot_reconstruction_limiters.jl")

function comparison_table(profile; N=512)
    name, f, primitive = profile
    xface = (1:N) ./ N
    println("\n", name)
    @printf("%-30s %14s %14s %14s\n", "reconstruction", "Linf left", "L1 left", "jump L1")
    for recon in RECONSTRUCTIONS
        u = sample_profile(recon, f, primitive, N)
        left, right = reconstruct_periodic_faces(recon, u)
        error = abs.(left .- f.(xface))
        @printf("%-30s %14.6e %14.6e %14.6e\n", recon_label(recon),
            maximum(error), sum(error) / N, sum(abs.(right .- left)) / N)
    end
    return nothing
end

function main()
    foreach(comparison_table, PROFILES)
    written = make_reconstruction_plots()
    isempty(written) || @info "Wrote reconstruction plots" written
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
