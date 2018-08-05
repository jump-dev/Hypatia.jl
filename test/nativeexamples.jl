
@testset "envelope example" begin
    alf = build_envelope(2, 5, 1, 5, use_data=true)
    Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ -25.502777 rtol=1e-4
    @test Alfonso.get_pobj(alf) ≈ -25.502777 rtol=1e-4
end

@testset "lp example" begin
    alf = build_lp(500, 1000, use_data=true)
    Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ 2055.807 rtol=1e-4
    @test Alfonso.get_pobj(alf) ≈ 2055.807 rtol=1e-4
end

@testset "namedpoly examples" begin
    @testset "Goldstein Price" begin
        alf = build_namedpoly(:goldsteinprice, 7)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ 3 rtol=1e-4
        @test Alfonso.get_pobj(alf) ≈ 3 rtol=1e-4
    end

    @testset "Robinson" begin
        alf = build_namedpoly(:robinson, 8)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ 0.814814 rtol=1e-4
        @test Alfonso.get_pobj(alf) ≈ 0.814814 rtol=1e-4
    end

    @testset "Lotka Volterra" begin
        alf = build_namedpoly(:lotkavolterra, 3)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ -20.8 rtol=1e-4
        @test Alfonso.get_pobj(alf) ≈ -20.8 rtol=1e-4
    end
end
