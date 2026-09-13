const BENCHMARK_SOURCE = isempty(ARGS) ? dirname(@__DIR__) : only(ARGS)
const BENCHMARK_ENTRY = isfile(joinpath(BENCHMARK_SOURCE, "src", "Reconstruction.jl")) ?
    joinpath(BENCHMARK_SOURCE, "src", "Reconstruction.jl") : joinpath(BENCHMARK_SOURCE, "Reconstruction.jl")
include(BENCHMARK_ENTRY)
if isdefined(@__MODULE__, :ReconstructionLimiters)
    using .ReconstructionLimiters
else
    using .Reconstruction
end
using Statistics, Printf, Random

@noinline function direct_faces!(l, r, recon, u)
    n = Val(stencil_size(recon))
    @inbounds for i in eachindex(l)
        l[i], r[i] = face(recon, ntuple(k -> u[i + k - 1], n)...)
    end
    return nothing
end

function measure_sweep(f; repetitions=100, samples=15)
    f()
    bytes = @allocated f()
    times = map(1:samples) do _
        (@elapsed for _ in 1:repetitions
            f()
        end) / repetitions
    end
    return (microseconds=1e6 * median(times), bytes=bytes)
end

function periodic_samples(recon, input, profile, N)
    radius = left_stencil_size(recon) ÷ 2
    dx = 1 / N
    rng = MersenneTwister(129)
    raw = [begin
        x = (i - 1) / N
        if profile == "rough"
            randn(rng)
        else
            a, b = input == "averages" ? (sinc(dx), sinc(3dx)) : (1.0, 1.0)
            jump = if profile != "mixed"
                0.0
            elseif input == "averages"
                max(0.0, min(x + dx / 2, 0.7) - max(x - dx / 2, 0.3)) / dx
            else
                0.3 <= x < 0.7 ? 1.0 : 0.0
            end
            a*sin(2π*x) + 0.2b*cos(6π*x) + jump
        end
    end for i in 1:N]
    return [raw[mod1(i - radius, N)] for i in 1:(N + 2radius + 1)]
end
