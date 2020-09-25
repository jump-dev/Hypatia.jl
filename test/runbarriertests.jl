#=
run barrier tests
=#

using Test
using Printf
include(joinpath(@__DIR__, "barrier.jl"))

barrier_test_names = [
    "nonnegative",
    "epinorminf",
    "epinormeucl",
    "epipersquare",
    "episumperentropy",
    "hypoperlog",
    "power",
    "hypogeomean",
    "hypopowermean",
    "epinormspectral",
    "linmatrixineq", # NOTE failing on Julia v1.5.1
    "possemideftri",
    "possemideftrisparse",
    "doublynonnegative",
    "matrixepipersquare",
    "hypoperlogdettri",
    "hyporootdettri",
    "wsosinterpnonnegative",
    "wsosinterppossemideftri",
    "wsosinterpepinormeucl",
    ]

real_types = [
    Float64,
    # Float32,
    # BigFloat,
    ]

@testset "barrier tests" begin
@testset "$name" for name in barrier_test_names
@testset "$T" for T in real_types
    println("$name: $T ...")
    test_time = @elapsed eval(Symbol("test_", name, "_barrier"))(T)
    @printf("%4.2f seconds\n", test_time)
end
end
end
;
