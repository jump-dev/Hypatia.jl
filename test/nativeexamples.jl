
@testset "envelope example" begin
    alf = build_envelope(2, 5, 1, 5, use_data=true)
    Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ -25.502777 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ -25.502777 rtol=1e-4
end

@testset "lp example" begin
    alf = build_lp(500, 1000, use_data=true)
    Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ 2055.807 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ 2055.807 rtol=1e-4
end

# TODO use known minimum values in boxes
@testset "namedpoly examples" begin
    @testset "Butcher" begin
        alf = build_namedpoly(:butcher, 2)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ -1.439334 rtol=1e-4
        @test Alfonso.get_dobj(alf) ≈ -1.439334 rtol=1e-4
    end

    @testset "Caprasse" begin
        alf = build_namedpoly(:caprasse, 4)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ -3.3207848 rtol=1e-4
        @test Alfonso.get_dobj(alf) ≈ -3.3207848 rtol=1e-4
    end

    @testset "Goldstein Price" begin
        alf = build_namedpoly(:goldsteinprice, 7)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ 3 rtol=1e-4
        @test Alfonso.get_dobj(alf) ≈ 3 rtol=1e-4
    end

    # @testset "Heart" begin
    #     alf = build_namedpoly(:heart, 2)
    #     Alfonso.solve!(alf)
    #     @test Alfonso.get_status(alf) == :Optimal
    #     @test Alfonso.get_pobj(alf) ≈ 3 rtol=1e-4
    #     @test Alfonso.get_dobj(alf) ≈ 3 rtol=1e-4
    # end

    @testset "Lotka Volterra" begin
        alf = build_namedpoly(:lotkavolterra, 3)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ -20.8 rtol=1e-4
        @test Alfonso.get_dobj(alf) ≈ -20.8 rtol=1e-4
    end

    # @testset "Magnetism-7" begin
    #     alf = build_namedpoly(:magnetism7, 2)
    #     Alfonso.solve!(alf)
    #     @test Alfonso.get_status(alf) == :Optimal
    #     @test Alfonso.get_pobj(alf) ≈ 3 rtol=1e-4
    #     @test Alfonso.get_dobj(alf) ≈ 3 rtol=1e-4
    # end

    @testset "Motzkin" begin
        alf = build_namedpoly(:motzkin, 12)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ 0 rtol=1e-4
        @test Alfonso.get_dobj(alf) ≈ 0 rtol=1e-4
    end

    @testset "Reaction-diffusion" begin
        alf = build_namedpoly(:reactiondiffusion, 4)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ -36.712707 rtol=1e-4
        @test Alfonso.get_dobj(alf) ≈ -36.712707 rtol=1e-4
    end

    @testset "Robinson" begin
        alf = build_namedpoly(:robinson, 8)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ 0.814814 rtol=1e-4
        @test Alfonso.get_dobj(alf) ≈ 0.814814 rtol=1e-4
    end

    @testset "Rosenbrock" begin
        alf = build_namedpoly(:rosenbrock, 8)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ 0.01039699 rtol=1e-4
        @test Alfonso.get_dobj(alf) ≈ 0.01039699 rtol=1e-4
    end

    @testset "Schwefel" begin
        alf = build_namedpoly(:schwefel, 6)
        Alfonso.solve!(alf)
        @test Alfonso.get_status(alf) == :Optimal
        @test Alfonso.get_pobj(alf) ≈ 0.0029282 rtol=1e-4
        @test Alfonso.get_dobj(alf) ≈ 0.0029282 rtol=1e-4
    end
end
