#=
run subset of tests
=#

using Test
using Printf

test_files = [
    # "modelutilities",
    # "barrier",
    "native",
    # "moi",
    # "cblib",
    # "examples",
    # NOTE require optional dependencies:
    # "pardiso",
    # "hsl",
    ]

@info("starting all tests")
@testset "all tests" begin
all_test_time = @elapsed @testset "$t" for t in test_files
    @info("starting $t tests")
    test_time = @elapsed include("run$(t)tests.jl")
    @info("finished $t tests in $(@sprintf("%8.2e seconds", test_time))")
    println()
    flush(stdout); flush(stderr)
end
@info("finished all tests in $(@sprintf("%8.2e seconds", all_test_time))")
println()
end
;
