# Plotting entry point. Run correctness tests with Pkg.test().
include("common.jl")

# This script writes figures to files and does not need a GUI window.
get!(ENV, "GKSwstype", "100")
const HAS_PLOTS = try
    @eval using Plots: plot, plot!, savefig
    true
catch err
    @warn "Plots.jl is not available; skipping plotting diagnostics." exception=(err, catch_backtrace())
    false
end

function make_reconstruction_plots(; N=96, outdir=joinpath(@__DIR__, "reconstruction_plots"))
    HAS_PLOTS || return String[]

    mkpath(outdir)

    xcell = ((0:N-1) .+ 0.5) ./ N
    xface = (1:N) ./ N

    written = String[]

    for (name, f, primitive) in PROFILES

        p = plot(
            xcell,
            f.(xcell);
            label="exact point values",
            linewidth=3,
            xlabel="x",
            ylabel="reconstructed left state",
            title="Left state at faces: $name",
            legend=:outerright,
        )

        for recon in PLOT_RECONS
            u = sample_profile(recon, f, primitive, N)
            l, _ = reconstruct_periodic_faces(recon, u)
            plot!(p, xface, l; label=recon_label(recon), linewidth=1.5)
        end

        path_left = joinpath(outdir, "$(name)_left_states.png")
        savefig(p, path_left)
        push!(written, path_left)

        q = plot(
            xface,
            zeros(N);
            label="zero",
            linewidth=2,
            xlabel="x",
            ylabel="|uR - uL|",
            title="Face jump diagnostic: $name",
            legend=:outerright,
        )

        for recon in PLOT_RECONS
            u = sample_profile(recon, f, primitive, N)
            l, r = reconstruct_periodic_faces(recon, u)
            plot!(q, xface, abs.(r .- l); label=recon_label(recon), linewidth=1.5)
        end

        path_jump = joinpath(outdir, "$(name)_face_jumps.png")
        savefig(q, path_jump)
        push!(written, path_jump)

        # A smooth profile provides an exact derivative for comparison.
        if name == "sine"
            cplot = plot(xcell, f.(xcell); label="exact", xlabel="x",
                ylabel="center value", title="Cell-center values: sine", legend=:outerright)
            dplot = plot(xcell, 2π .* cos.(2π .* xcell); label="exact", xlabel="x",
                ylabel="du/dx", title="Cell-center derivatives: sine", legend=:outerright)
            for recon in PLOT_RECONS
                has_center(recon) || continue
                u = sample_profile(recon, f, primitive, N)
                c, d = reconstruct_periodic_centers(recon, u)
                plot!(cplot, xcell, c; label=recon_label(recon))
                plot!(dplot, xcell, d; label=recon_label(recon))
            end
            for (figure, suffix) in ((cplot, "center_values"), (dplot, "center_derivatives"))
                path = joinpath(outdir, "sine_$(suffix).png")
                savefig(figure, path)
                push!(written, path)
            end
        end
    end

    return written
end

if abspath(PROGRAM_FILE) == @__FILE__
    written = make_reconstruction_plots()
    if !isempty(written)
        @info "Wrote reconstruction diagnostic plots" written
    end
end
