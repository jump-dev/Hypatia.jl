#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Test

@info("starting minimal tests")
@testset "minimal tests" begin
    include(joinpath(@__DIR__, "runmodelutilitiestests.jl"))
    include(joinpath(@__DIR__, "runbarriertests.jl"))
    include(joinpath(@__DIR__, "runnativetests.jl"))
    include(joinpath(@__DIR__, "runmoitests.jl"))
end
