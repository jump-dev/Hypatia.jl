#=
Copyright 2019, Chris Coey and contributors

MOI wrapper Hypatia cone tests
=#

using Test
using LinearAlgebra
import MathOptInterface
const MOI = MathOptInterface
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones

function test_moi_cones(T::Type{<:Real})
    # MOI predefined cones

    @testset "NormInfinityCone" begin
        moi_cone = MOI.NormInfinityCone(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiNormInf{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test !CO.use_dual(hyp_cone)
    end

    @testset "NormOneCone" begin
        moi_cone = MOI.NormOneCone(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiNormInf{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test CO.use_dual(hyp_cone)
    end

    @testset "SecondOrderCone" begin
        moi_cone = MOI.SecondOrderCone(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiNormEucl{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test !CO.use_dual(hyp_cone)
    end

    @testset "RotatedSecondOrderCone" begin
        moi_cone = MOI.RotatedSecondOrderCone(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiPerSquare{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test !CO.use_dual(hyp_cone)
    end

    @testset "ExponentialCone" begin
        moi_cone = MOI.ExponentialCone()
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoPerLog{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test !CO.use_dual(hyp_cone)
    end

    @testset "DualExponentialCone" begin
        moi_cone = MOI.DualExponentialCone()
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoPerLog{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test CO.use_dual(hyp_cone)
    end

    @testset "PowerCone" begin
        iT5 = inv(T(5))
        moi_cone = MOI.PowerCone{T}(iT5)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.Power{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test !CO.use_dual(hyp_cone)
        @test hyp_cone.alpha == T[iT5, 1 - iT5]
    end

    @testset "DualPowerCone" begin
        iT5 = inv(T(5))
        moi_cone = MOI.DualPowerCone{T}(iT5)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.Power{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test CO.use_dual(hyp_cone)
        @test hyp_cone.alpha == T[iT5, 1 - iT5]
    end

    @testset "GeometricMeanCone" begin
        moi_cone = MOI.GeometricMeanCone(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoGeomean{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test !CO.use_dual(hyp_cone)
        iT2 = inv(T(2))
        @test hyp_cone.alpha == T[iT2, iT2]
    end

    @testset "PositiveSemidefiniteConeTriangle" begin
        moi_cone = MOI.PositiveSemidefiniteConeTriangle(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.PosSemidefTri{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 6
        @test !CO.use_dual(hyp_cone)
    end

    @testset "LogDetConeTriangle" begin
        moi_cone = MOI.LogDetConeTriangle(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoPerLogdetTri{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 8
        @test !CO.use_dual(hyp_cone)
    end

    @testset "RootDetConeTriangle" begin
        moi_cone = MOI.RootDetConeTriangle(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoRootdetTri{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 7
        @test !CO.use_dual(hyp_cone)
    end

    # Hypatia predefined cones

    @testset "Nonnegative" begin
        moi_cone = HYP.NonnegativeCone{T}(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.Nonnegative{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
    end

    @testset "EpiNormInfinity" begin
        moi_cone = HYP.EpiNormInfinityCone{T, T}(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiNormInf{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3

        moi_cone = HYP.EpiNormInfinityCone{T, Complex{T}}(5)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiNormInf{T, Complex{T}}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 5
    end

    @testset "EpiNormEucl" begin
        moi_cone = HYP.EpiNormEuclCone{T}(3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiNormEucl{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
    end

    @testset "EpiPerSquare" begin
        moi_cone = HYP.EpiPerSquareCone{T}(4)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiPerSquare{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 4
    end

    @testset "Power" begin
        alpha = rand(T, 2)
        alpha ./= sum(alpha)
        moi_cone = HYP.PowerCone{T}(alpha, 3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.Power{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 5
        @test hyp_cone.alpha == alpha
    end

    @testset "HypoPerLog" begin
        moi_cone = HYP.HypoPerLogCone{T}(4)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoPerLog{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 4
    end

    @testset "EpiPerExp" begin
        moi_cone = HYP.EpiPerExpCone{T}()
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiPerExp{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
    end

    @testset "EpiSumPerEntropy" begin
        moi_cone = HYP.EpiSumPerEntropyCone{T}(4)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiSumPerEntropy{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 4
    end

    @testset "HypoGeomean" begin
        alpha = rand(T, 2)
        alpha ./= sum(alpha)
        moi_cone = HYP.HypoGeomeanCone{T}(alpha)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoGeomean{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test hyp_cone.alpha == alpha
    end

    @testset "EpiNormSpectral" begin
        moi_cone = HYP.EpiNormSpectralCone{T, T}(2, 3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiNormSpectral{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 7

        moi_cone = HYP.EpiNormSpectralCone{T, Complex{T}}(2, 3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.EpiNormSpectral{T, Complex{T}}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 13
    end

    @testset "MatrixEpiPerSquare" begin
        moi_cone = HYP.MatrixEpiPerSquareCone{T, T}(2, 3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.MatrixEpiPerSquare{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 10

        moi_cone = HYP.MatrixEpiPerSquareCone{T, Complex{T}}(2, 3)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.MatrixEpiPerSquare{T, Complex{T}}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 17
    end

    @testset "LinMatrixIneq" begin
        As = [Symmetric(Matrix(one(T) * I, 2, 2)), Hermitian(Complex{T}[1 0; 0 -1])]
        moi_cone = HYP.LinMatrixIneqCone{T}(As)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.LinMatrixIneq{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 2
        @test hyp_cone.As == As
    end

    @testset "PosSemidefTri" begin
        moi_cone = HYP.PosSemidefTriCone{T, T}(6)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.PosSemidefTri{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 6

        moi_cone = HYP.PosSemidefTriCone{T, Complex{T}}(9)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.PosSemidefTri{T, Complex{T}}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 9
    end

    @testset "HypoPerLogdetTri" begin
        moi_cone = HYP.HypoPerLogdetTriCone{T, T}(8)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoPerLogdetTri{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 8

        moi_cone = HYP.HypoPerLogdetTriCone{T, Complex{T}}(11)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoPerLogdetTri{T, Complex{T}}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 11
    end

    @testset "HypoRootdetTri" begin
        moi_cone = HYP.HypoRootdetTriCone{T, T}(7)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoRootdetTri{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 7

        moi_cone = HYP.HypoRootdetTriCone{T, Complex{T}}(10)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.HypoRootdetTri{T, Complex{T}}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 10
    end

    @testset "WSOSInterpNonnegative" begin
        Ps = [rand(T, 3, 2), rand(T, 3, 1)]
        moi_cone = HYP.WSOSInterpNonnegativeCone{T, T}(3, Ps)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.WSOSInterpNonnegative{T, T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 3
        @test hyp_cone.Ps == Ps

        Ps = [rand(Complex{T}, 4, 3), rand(Complex{T}, 4, 2)]
        moi_cone = HYP.WSOSInterpNonnegativeCone{T, Complex{T}}(4, Ps)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.WSOSInterpNonnegative{T, Complex{T}}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 4
        @test hyp_cone.Ps == Ps
    end

    @testset "WSOSInterpPosSemidefTri" begin
        Ps = [rand(T, 3, 2), rand(T, 3, 1)]
        moi_cone = HYP.WSOSInterpPosSemidefTriCone{T}(2, 3, Ps)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.WSOSInterpPosSemidefTri{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 9
        @test hyp_cone.Ps == Ps
    end

    @testset "WSOSInterpEpiNormEucl" begin
        Ps = [rand(T, 3, 2), rand(T, 3, 1)]
        moi_cone = HYP.WSOSInterpEpiNormEuclCone{T}(2, 3, Ps)
        hyp_cone = HYP.cone_from_moi(T, moi_cone)
        @test hyp_cone isa CO.WSOSInterpEpiNormEucl{T}
        @test MOI.dimension(moi_cone) == CO.dimension(hyp_cone) == 6
        @test hyp_cone.Ps == Ps
    end

    return
end
