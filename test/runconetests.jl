#=
run barrier tests
=#

using Test
using Printf
import Hypatia.Cones
include(joinpath(@__DIR__, "cone.jl"))

function cone_types(T::Type{<:Real})
    cones_T = [
        # Cones.Nonnegative{T},
        # Cones.PosSemidefTri{T, T},
        # Cones.PosSemidefTri{T, Complex{T}},
        # Cones.DoublyNonnegativeTri{T},
        Cones.LinMatrixIneq{T},
        # Cones.EpiNormInf{T, T},
        # Cones.EpiNormInf{T, Complex{T}},
        # Cones.EpiNormEucl{T},
        # Cones.EpiPerSquare{T},
        # Cones.EpiNormSpectral{T, T},
        # Cones.EpiNormSpectral{T, Complex{T}},
        # Cones.MatrixEpiPerSquare{T, T},
        # Cones.MatrixEpiPerSquare{T, Complex{T}},
        # Cones.GeneralizedPower{T},
        # Cones.HypoPowerMean{T},
        # Cones.HypoGeoMean{T},
        # Cones.HypoRootdetTri{T, T},
        # Cones.HypoRootdetTri{T, Complex{T}},
        # Cones.HypoPerLog{T},
        # Cones.HypoPerLogdetTri{T, T},
        # Cones.HypoPerLogdetTri{T, Complex{T}},
        # Cones.EpiPerEntropy{T},
        # Cones.EpiPerTraceEntropyTri{T},
        # Cones.EpiRelEntropy{T},
        # Cones.EpiTraceRelEntropyTri{T}, # TODO tighten tol on test_barrier
        # Cones.WSOSInterpNonnegative{T, T},
        # # Cones.WSOSInterpNonnegative{T, Complex{T}}, # TODO
        # Cones.WSOSInterpPosSemidefTri{T},
        # Cones.WSOSInterpEpiNormEucl{T},
        # Cones.WSOSInterpEpiNormOne{T},
        ]
    if T <: LinearAlgebra.BlasReal
        append!(cones_T, [
            # Cones.PosSemidefTriSparse{T, T},
            # Cones.PosSemidefTriSparse{T, Complex{T}},
            ])
    end
    return cones_T
end

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

println("\nstarting time/allocation measurements")
@testset "allocation tests" begin
real_types = [
    Float64,
    # Float32,
    # BigFloat,
    ]
@testset "$cone" for T in real_types, cone in cone_types(T)
    println("\n$cone")
    test_time = @elapsed show_time_alloc(cone)
    @printf("%8.2e seconds\n", test_time)
end
println()
end

end
;
