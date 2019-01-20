#=
Copyright 2018, Chris Coey and contributors

functions and types for model data
=#

module Models

abstract type Model end

include("linearobjconic.jl")
# include("quadraticobjconic.jl")
# include("nonlinearobjconic.jl")

end
