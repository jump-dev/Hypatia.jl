#=
run subset of tests
=#

using Test
using Printf

test_files = [
    "modelutilities",
    "barrier",
    # "native",
    # "moi",
    # "cblib",
    # "examples",
    # NOTE require optional dependencies:
    # "pardiso",
    # "hsl",
    ]

@info("starting all tests")
@testset "all tests" begin
@testset "$t" for t in test_files
    @info("starting $t tests")
    test_time = @elapsed include("run$(t)tests.jl")
    @info("finished $t tests in $(@sprintf("%4.2f seconds", test_time))")
end
end
;
