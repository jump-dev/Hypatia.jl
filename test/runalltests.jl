#=
Copyright 2019, Chris Coey and contributors
=#

using Test

@info("starting Hypatia all tests")

@testset "Hypatia all tests" begin
    # include(joinpath(@__DIR__, "runinterptests.jl"))
    include(joinpath(@__DIR__, "runbarriertests.jl"))
    include(joinpath(@__DIR__, "runnativetests.jl"))
    include(joinpath(@__DIR__, "runmoitests.jl"))
    include(joinpath(@__DIR__, "runnativeexamplestests.jl"))
    include(joinpath(@__DIR__, "runjumpexamplestests.jl"))
end
