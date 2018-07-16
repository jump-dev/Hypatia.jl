
__precompile__()

module Alfonso
    import MathOptInterface
    const MOI = MathOptInterface

    include("optimizer.jl")
    include("model.jl")
end
