
module Alfonso
    import MathOptInterface
    const MOI = MathOptInterface
    using Printf
    using SparseArrays
    using LinearAlgebra
    import FFTW
    import Combinatorics

    include("interpfuns.jl")
    include("barriers.jl")
    include("optimizer.jl")
    include("algorithm.jl")

    export cheb2_data, padua_data, approxfekete_data
    export AlfonsoOptimizer
end
