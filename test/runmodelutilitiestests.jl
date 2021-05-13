#=
run model utilities tests
=#

using Test
using Printf
include(joinpath(@__DIR__, "modelutilities.jl"))

real_types = [
    Float64,
    Float32,
    BigFloat,
    ]

@testset "model utilities tests" begin
@testset "$T" for T in real_types
    println("$T ...")
    test_time = @elapsed begin
        test_svec_conversion(T)
        test_fekete_sample(T)
        test_cheb2_w(T)
    end
    @printf("%8.2e seconds\n", test_time)
end
end
;
