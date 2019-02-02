#=
Copyright 2018, Chris Coey and contributors

functions and caches for solving specially-structured linear systems
=#

module LinearSystems

using LinearAlgebra
# using LinearAlgebra: BlasInt
# include("lapack.jl")

# import IterativeSolvers

import Hypatia.Cones
import Hypatia.Models

abstract type LinearSystemSolver end

# include("naive.jl")
include("chol_chol.jl")
# include("qr_qr.jl")
# include("qr_chol.jl")

end
