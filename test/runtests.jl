using Test
using Reconstruction
using Random

const RL = Reconstruction
const CWENO_WEIGHT_STRATEGIES = (JSWeights(), ZWeights(), ZWeights(; power=1))

const RECONS = (
    Godunov(),
    WENO3(),
    WENOZ(),
    CWENO5(),
    CWENO5(; input=PointValues()),
    CWENO3(),
    CWENO3(; input=PointValues()),
    CWENO3(; weights=ZWeights()),
    CWENO3(; weights=ZWeights(), input=PointValues()),
    CWENO5(; weights=ZWeights()),
    CWENO5(; weights=ZWeights(), input=PointValues()),
    MP5(),
    MinmodLimiter(),
    GeneralizedMinmodLimiter(),
    VanLeerLimiter(),
    VanAlbadaLimiter(),
    MonotonizedCentralLimiter(),
    SuperbeeLimiter(),
)

include("runtests_interfaces.jl")
include("runtests_cell_polynomials.jl")
include("runtests_cweno_z.jl")
include("runtests_cweno_reference.jl")
include("runtests_weno.jl")
include("runtests_mp5_reference.jl")
include("runtests_examples.jl")
