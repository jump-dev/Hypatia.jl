#=
Copyright 2019, Chris Coey and contributors

MOI wrapper Hypatia cone tests
=#

using Test
import MathOptInterface
const MOI = MathOptInterface
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones

function test_moi_cones(T::Type{<:Real})
    @testset "Nonnegative" begin
        cone = HYP.NonnegativeCone{T}(3)
        @test MOI.dimension(cone) == 3
        @test HYP.cone_from_moi(T, cone) isa CO.Nonnegative{T}
    end

    # TODO others

    return
end
