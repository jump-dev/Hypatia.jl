#=
run barrier tests
=#

using Test
using Printf
include(joinpath(@__DIR__, "barrier.jl"))

barrier_test_names = [
    # "nonnegative",
    # "epinorminf",
    # "epinormeucl",
    # "epipersquare",
    # "episumperentropy",
    "hypoperlog",
    # "power",
    # "hypopowermean",
    # "hypogeomean",
    # "epinormspectral",
    # "linmatrixineq", # NOTE failing on Julia v1.5.1 with ForwardDiff or BigFloat
    # "possemideftri",
    # "possemideftrisparse",
    # "doublynonnegativetri",
    # "matrixepipersquare",
    "hypoperlogdettri",
    # "hyporootdettri",
    # "wsosinterpnonnegative",
    # "wsosinterpepinormone",
    # "wsosinterpepinormeucl",
    # "wsosinterppossemideftri",
    ]

real_types = [
    Float64,
    Float32,
    BigFloat,
    ]

@testset "barrier tests" begin
@testset "$name" for name in barrier_test_names
@testset "$T" for T in real_types
    println("$name: $T ...")
    test_time = @elapsed eval(Symbol("test_", name, "_barrier"))(T)
    @printf("%8.2e seconds\n", test_time)
end
end
end
;
