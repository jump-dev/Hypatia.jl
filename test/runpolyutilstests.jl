#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
run polynomial utilities tests
=#

using Test
using Printf
include(joinpath(@__DIR__, "polyutils.jl"))

real_types = [Float64, Float32, BigFloat]

@testset "polynomial utilities tests" begin
    @testset "$T" for T in real_types
        println("$T ...")
        test_time = @elapsed begin
            test_interp_domain(T)
            test_complex_interp(T)
            test_quadrature(T)
        end
        @printf("%8.2e seconds\n", test_time)
    end
end;
