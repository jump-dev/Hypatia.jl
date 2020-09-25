#=
run model utilities tests
=#

using Printf
include(joinpath(@__DIR__, "modelutilities.jl"))

real_types = [
    Float64,
    Float32,
    BigFloat,
    ]

@info("starting model utilities tests")
@testset "model utilities tests" begin
@testset "$T" for T in real_types
    println("$T ...")
    test_time = @elapsed begin
        test_svec_conversion(T)
        test_fekete_sample(T)
        test_cheb2_w(T)
        test_recover_lagrange_polys(T)
        test_recover_cheb_polys(T)
    end
    @printf("%4.2f seconds\n", test_time)
end
end
