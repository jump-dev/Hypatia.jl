
module Alfonso
    import MathOptInterface
    using Printf
    using SparseArrays
    using LinearAlgebra
    import FFTW
    import Combinatorics

    const MOI = MathOptInterface

    include("interpolation.jl")
    include("barriers.jl")
    include("optimizer.jl")
    include("algorithm.jl")

    export cheb2_data, padua_data, approxfekete_data
    export ConeData, NonnegData, SumOfSqrData
end
