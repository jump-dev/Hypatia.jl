#=
run subset of tests
=#

using Test
using Printf
using Hypatia

test_files = [
    "modelutilities",
    "barrier",
    "native",
    "moi",
    "examples",
    # NOTE require optional dependencies:
    # "pardiso",
    # "hsl",
    ]

println()
@info("starting all tests")
println()
@testset "all tests" begin
all_test_time = @elapsed for t in test_files
    @info("starting $t tests")
    test_time = @elapsed include("run$(t)tests.jl")
    @info("finished $t tests in $(@sprintf("%8.2e seconds", test_time))")
    println()
    flush(stdout); flush(stderr)
end
@info("finished all tests in $(@sprintf("%8.2e seconds", all_test_time))")
end
println()
;
