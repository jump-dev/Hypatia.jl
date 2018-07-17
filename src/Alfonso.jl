
__precompile__()

module Alfonso
    import MathOptInterface
    const MOI = MathOptInterface

    include("optimizer.jl")
    # include("cones.jl")
    include("algorithm.jl")
end
