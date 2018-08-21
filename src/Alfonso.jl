
module Alfonso
    using Printf
    using SparseArrays
    using LinearAlgebra

    include("interpolation.jl")
    include("primitivecones.jl")
    include("cone.jl")
    include("nativeinterface.jl")
    include("mathoptinterface.jl")
end
