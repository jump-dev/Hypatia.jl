#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

import Requires
@show "init.jl"
function __init__()
    Requires.@require Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" include(joinpath(@__DIR__(), "linearalgebra", "Pardiso.jl"))
end