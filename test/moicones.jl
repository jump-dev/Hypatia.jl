#=
tests for MOI wrapper
=#

using Test
using LinearAlgebra
import SparseArrays
import Random
import MathOptInterface
const MOI = MathOptInterface
import Hypatia
import Hypatia.Cones

function test_moi_cones(T::Type{<:Real})
    # MOI predefined cones

    @testset "PositiveSemidefiniteConeTriangle" begin
        moi_cone = MOI.PositiveSemidefiniteConeTriangle(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.PosSemidefTri{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 6
        @test !Cones.use_dual_barrier(hyp_cone)
    end

    @testset "NormInfinityCone" begin
        moi_cone = MOI.NormInfinityCone(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiNormInf{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test !Cones.use_dual_barrier(hyp_cone)
    end

    @testset "NormOneCone" begin
        moi_cone = MOI.NormOneCone(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiNormInf{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test Cones.use_dual_barrier(hyp_cone)
    end

    @testset "SecondOrderCone" begin
        moi_cone = MOI.SecondOrderCone(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiNormEucl{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test !Cones.use_dual_barrier(hyp_cone)
    end

    @testset "RotatedSecondOrderCone" begin
        moi_cone = MOI.RotatedSecondOrderCone(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiPerSquare{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test !Cones.use_dual_barrier(hyp_cone)
    end

    @testset "NormSpectralCone" begin
        moi_cone = MOI.NormSpectralCone(2, 3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiNormSpectral{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 7
        @test !Cones.use_dual_barrier(hyp_cone)
    end

    @testset "NormNuclearCone" begin
        moi_cone = MOI.NormNuclearCone(2, 3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiNormSpectral{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 7
        @test Cones.use_dual_barrier(hyp_cone)
    end

    @testset "PowerCone" begin
        iT5 = inv(T(5))
        moi_cone = MOI.PowerCone{T}(iT5)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.GeneralizedPower{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test !Cones.use_dual_barrier(hyp_cone)
        @test hyp_cone.α == T[iT5, 1 - iT5]
    end

    @testset "DualPowerCone" begin
        iT5 = inv(T(5))
        moi_cone = MOI.DualPowerCone{T}(iT5)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.GeneralizedPower{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test Cones.use_dual_barrier(hyp_cone)
        @test hyp_cone.α == T[iT5, 1 - iT5]
    end

    @testset "GeometricMeanCone" begin
        moi_cone = MOI.GeometricMeanCone(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoGeoMean{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test !Cones.use_dual_barrier(hyp_cone)
    end

    @testset "RootDetConeTriangle" begin
        moi_cone = MOI.RootDetConeTriangle(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoRootdetTri{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 7
        @test !Cones.use_dual_barrier(hyp_cone)
    end

    @testset "ExponentialCone" begin
        moi_cone = MOI.ExponentialCone()
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoPerLog{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test !Cones.use_dual_barrier(hyp_cone)
    end

    @testset "DualExponentialCone" begin
        moi_cone = MOI.DualExponentialCone()
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoPerLog{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test Cones.use_dual_barrier(hyp_cone)
    end

    @testset "LogDetConeTriangle" begin
        moi_cone = MOI.LogDetConeTriangle(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoPerLogdetTri{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 8
        @test !Cones.use_dual_barrier(hyp_cone)
    end

    @testset "RelativeEntropyCone" begin
        moi_cone = MOI.RelativeEntropyCone(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiRelEntropy{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test !Cones.use_dual_barrier(hyp_cone)
    end

    # Hypatia predefined cones

    @testset "Nonnegative" begin
        moi_cone = Hypatia.NonnegativeCone{T}(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.Nonnegative{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
    end

    @testset "PosSemidefTri" begin
        moi_cone = Hypatia.PosSemidefTriCone{T, T}(6)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.PosSemidefTri{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 6

        moi_cone = Hypatia.PosSemidefTriCone{T, Complex{T}}(9)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.PosSemidefTri{T, Complex{T}}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 9
    end

    @testset "DoublyNonnegativeTri" begin
        moi_cone = Hypatia.DoublyNonnegativeTriCone{T}(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.DoublyNonnegativeTri{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
    end

    @testset "PosSemidefTriSparse" begin
        for (psdsparseimpl, realtypes) in Cones.PSDSparseImplList
            (T <: realtypes) || continue

            Random.seed!(1)
            side = 5
            (row_idxs, col_idxs, _) = SparseArrays.findnz(
                tril!(SparseArrays.sprand(Bool, side, side, 0.3)) + I)

            moi_cone = Hypatia.PosSemidefTriSparseCone{psdsparseimpl, T, T}(
                side, row_idxs, col_idxs)
            hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
            @test hyp_cone isa Cones.PosSemidefTriSparse{psdsparseimpl, T, T}
            @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) ==
                length(row_idxs)

            moi_cone = Hypatia.PosSemidefTriSparseCone{psdsparseimpl, T, Complex{T}}(
                side, row_idxs, col_idxs)
            hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
            @test hyp_cone isa
                Cones.PosSemidefTriSparse{psdsparseimpl, T, Complex{T}}
            @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) ==
                2 * length(row_idxs) - side
        end
    end

    @testset "LinMatrixIneq" begin
        As = [Symmetric(Matrix(one(T) * I, 2, 2)),
            Hermitian(Complex{T}[1 0; 0 -1])]
        moi_cone = Hypatia.LinMatrixIneqCone{T}(As)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.LinMatrixIneq{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 2
        @test hyp_cone.As == As
    end

    @testset "EpiNormInf" begin
        moi_cone = Hypatia.EpiNormInfCone{T, T}(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiNormInf{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3

        moi_cone = Hypatia.EpiNormInfCone{T, Complex{T}}(5)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiNormInf{T, Complex{T}}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 5
    end

    @testset "EpiNormEucl" begin
        moi_cone = Hypatia.EpiNormEuclCone{T}(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiNormEucl{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
    end

    @testset "EpiPerSquare" begin
        moi_cone = Hypatia.EpiPerSquareCone{T}(4)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiPerSquare{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 4
    end

    @testset "EpiNormSpectral" begin
        moi_cone = Hypatia.EpiNormSpectralCone{T, T}(2, 3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiNormSpectral{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 7

        moi_cone = Hypatia.EpiNormSpectralCone{T, Complex{T}}(2, 3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiNormSpectral{T, Complex{T}}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 13
    end

    @testset "MatrixEpiPerSquare" begin
        moi_cone = Hypatia.MatrixEpiPerSquareCone{T, T}(2, 3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.MatrixEpiPerSquare{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 10

        moi_cone = Hypatia.MatrixEpiPerSquareCone{T, Complex{T}}(2, 3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.MatrixEpiPerSquare{T, Complex{T}}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 17
    end

    @testset "GeneralizedPower" begin
        α = rand(T, 2)
        α ./= sum(α)
        moi_cone = Hypatia.GeneralizedPowerCone{T}(α, 3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.GeneralizedPower{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 5
        @test hyp_cone.α == α
    end

    @testset "HypoPowerMean" begin
        α = rand(T, 2)
        α ./= sum(α)
        moi_cone = Hypatia.HypoPowerMeanCone{T}(α)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoPowerMean{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test hyp_cone.α == α
    end

    @testset "HypoGeoMean" begin
        moi_cone = Hypatia.HypoGeoMeanCone{T}(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoGeoMean{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
    end

    @testset "HypoRootdetTri" begin
        moi_cone = Hypatia.HypoRootdetTriCone{T, T}(7)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoRootdetTri{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 7

        moi_cone = Hypatia.HypoRootdetTriCone{T, Complex{T}}(10)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoRootdetTri{T, Complex{T}}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 10
    end

    @testset "HypoPerLog" begin
        moi_cone = Hypatia.HypoPerLogCone{T}(4)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoPerLog{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 4
    end

    @testset "HypoPerLogdetTri" begin
        moi_cone = Hypatia.HypoPerLogdetTriCone{T, T}(8)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoPerLogdetTri{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 8

        moi_cone = Hypatia.HypoPerLogdetTriCone{T, Complex{T}}(11)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.HypoPerLogdetTri{T, Complex{T}}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 11
    end

    @testset "EpiPerSepSpectral" begin
        h_fun = Cones.NegEntropySSF()
        QTs = [
            Cones.VectorCSqr{T},
            Cones.MatrixCSqr{T, T},
            Cones.MatrixCSqr{T, Complex{T}},
            ]
        for QT in QTs
            moi_cone = Hypatia.EpiPerSepSpectralCone{T}(h_fun, QT, 3)
            hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
            @test hyp_cone isa Cones.EpiPerSepSpectral{QT, T}
            @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone)
        end
    end

    @testset "EpiRelEntropy" begin
        moi_cone = Hypatia.EpiRelEntropyCone{T}(5)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiRelEntropy{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 5
    end

    @testset "EpiTrRelEntropyTriCone" begin
        moi_cone = Hypatia.EpiTrRelEntropyTriCone{T}(3)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.EpiTrRelEntropyTri{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
    end

    @testset "WSOSInterpNonnegative" begin
        Ps = [rand(T, 3, 2), rand(T, 3, 1)]
        moi_cone = Hypatia.WSOSInterpNonnegativeCone{T, T}(3, Ps)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.WSOSInterpNonnegative{T, T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 3
        @test hyp_cone.Ps == Ps

        Ps = [rand(Complex{T}, 4, 3), rand(Complex{T}, 4, 2)]
        moi_cone = Hypatia.WSOSInterpNonnegativeCone{T, Complex{T}}(4, Ps)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.WSOSInterpNonnegative{T, Complex{T}}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 4
        @test hyp_cone.Ps == Ps
    end

    @testset "WSOSInterpPosSemidefTri" begin
        Ps = [rand(T, 3, 2), rand(T, 3, 1)]
        moi_cone = Hypatia.WSOSInterpPosSemidefTriCone{T}(2, 3, Ps)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.WSOSInterpPosSemidefTri{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 9
        @test hyp_cone.Ps == Ps
    end

    @testset "WSOSInterpEpiNormOne" begin
        Ps = [rand(T, 3, 2), rand(T, 3, 1)]
        moi_cone = Hypatia.WSOSInterpEpiNormOneCone{T}(2, 3, Ps)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.WSOSInterpEpiNormOne{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 6
        @test hyp_cone.Ps == Ps
    end

    @testset "WSOSInterpEpiNormEucl" begin
        Ps = [rand(T, 3, 2), rand(T, 3, 1)]
        moi_cone = Hypatia.WSOSInterpEpiNormEuclCone{T}(2, 3, Ps)
        hyp_cone = Hypatia.cone_from_moi(T, moi_cone)
        @test hyp_cone isa Cones.WSOSInterpEpiNormEucl{T}
        @test MOI.dimension(moi_cone) == Cones.dimension(hyp_cone) == 6
        @test hyp_cone.Ps == Ps
    end

    return
end
