
using Alfonso
using Test


# TODO interpolation tests


# optimizer tests
include(joinpath(@__DIR__, "../examples/envelope/envelope.jl"))
@testset "envelope example" begin
    r = solve_envelope(2, 5, 1, 5, use_data=true)
    @test r.status == MOI.Success
    @test r.objval ≈ -25.502777 rtol=1e-4
    @test r.objbnd ≈ r.objval rtol=1e-4
end

include(joinpath(@__DIR__, "../examples/lp/lp.jl"))
@testset "lp example" begin
    r = solve_lp(500, 1000, use_data=true)
    @test r.status == MOI.Success
    @test r.objval ≈ 2055.807 rtol=1e-4
    @test r.objbnd ≈ r.objval rtol=1e-4
end

include(joinpath(@__DIR__, "../examples/namedpoly/namedpoly.jl"))
@testset "namedpoly examples" begin
    @testset "Goldstein Price" begin
        r = solve_namedpoly(:goldsteinprice, 7)
        # @test r.status == MOI.Success
        @test r.objval ≈ 3 rtol=1e-4
        @test r.objbnd ≈ r.objval rtol=1e-4
    end

    @testset "Robinson" begin
        r = solve_namedpoly(:robinson, 8)
        @test r.status == MOI.Success
        @test r.objval ≈ 0.814814 rtol=1e-4
        @test r.objbnd ≈ r.objval rtol=1e-4
    end

    @testset "Lotka Volterra" begin
        r = solve_namedpoly(:lotkavolterra, 3)
        @test r.status == MOI.Success
        @test r.objval ≈ -20.8 rtol=1e-4
        @test r.objbnd ≈ r.objval rtol=1e-4
    end
end
