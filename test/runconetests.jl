#=
run barrier tests
=#

using Test
using Printf
import Hypatia.Cones
include(joinpath(@__DIR__, "cone.jl"))

cone_types(T::Type{<:Real}) = [
    Cones.Nonnegative{T},
    Cones.EpiNormInf{T, T},
    Cones.EpiNormInf{T, Complex{T}},
    Cones.EpiNormEucl{T},
    Cones.EpiPerSquare{T},
    Cones.HypoPerLog{T},
    Cones.EpiPerEntropy{T},
    Cones.EpiRelEntropy{T},
    Cones.HypoGeoMean{T},
    Cones.HypoPowerMean{T},
    Cones.GeneralizedPower{T},
    ]

# barrier_test_names = [
#     "nonnegative",
#     "epinorminf",
#     "epinormeucl",
#     # "epiperentropy",
#     # "epipersquare",
#     # "epipertraceentropytri",
#     # "epitracerelentropytri",
#     # "epirelentropy",
#     # "hypoperlog",
#     # "power",
#     # "hypopowermean",
#     # "hypogeomean",
#     # "epinormspectral",
#     # "linmatrixineq", # NOTE failing on Julia v1.5.1 with ForwardDiff or BigFloat
#     # "possemideftri",
#     # "possemideftrisparse",
#     # "doublynonnegativetri",
#     # "matrixepipersquare",
#     # "hypoperlogdettri",
#     # "hyporootdettri",
#     # "epitracerelentropytri",
#     # "wsosinterpnonnegative",
#     # "wsosinterpepinormone",
#     # "wsosinterpepinormeucl",
#     # "wsosinterppossemideftri",
#     ]
#

@testset "cone tests" begin

println("starting oracle tests")
@testset "oracle tests" begin
real_types = [
    Float64,
    Float32,
    BigFloat,
    ]
@testset "$cone" for T in real_types, cone in cone_types(T)
    println("$cone")
    test_time = @elapsed test_oracles(cone)
    @printf("%8.2e seconds\n", test_time)
end
end

println("\nstarting barrier tests")
@testset "barrier tests" begin
real_types = [
    Float64,
    # Float32,
    # BigFloat,
    ]
@testset "$cone" for T in real_types, cone in cone_types(T)
    println("$cone")
    test_time = @elapsed test_barrier(cone)
    @printf("%8.2e seconds\n", test_time)
end
end

println("\nstarting allocation tests")
@testset "allocation tests" begin
real_types = [
    Float64,
    # Float32,
    # BigFloat,
    ]
@testset "$cone" for T in real_types, cone in cone_types(T)
    println("\n$cone")
    test_time = @elapsed allocs = get_allocs(cone)
    display(allocs)
    @printf("%8.2e seconds\n", test_time)
end
println()
end

end
;
