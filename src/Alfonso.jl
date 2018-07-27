
__precompile__()

module Alfonso
    import MathOptInterface
    const MOI = MathOptInterface
    using Printf
    using SparseArrays
    using LinearAlgebra
    import FFTW
    import Combinatorics

    include("barriers.jl")
    include("optimizer.jl")
    include("algorithm.jl")
    include("interpfuns.jl")

    export cheb2_data, padua_data
    export AlfonsoOptimizer
end
