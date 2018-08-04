
module Alfonso
    import MathOptInterface
    const MOI = MathOptInterface
    const MOIU = MOI.Utilities

    using Printf
    using SparseArrays
    using LinearAlgebra
    import FFTW
    import Combinatorics

    include("interpolation.jl")
    include("barriers.jl")
    include("optimizer.jl")
    include("algorithm.jl")

    export cheb2_data, padua_data, approxfekete_data
    export ConeData, NonnegData, SumOfSqrData
end
