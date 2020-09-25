#=
run subset of tests
=#

using Test

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
    test_time = @elapsed include("run$(t)tests.jl")
    println("finished $t tests")
    @printf("%4.2f seconds\n\n", test_time)
end
end
;
