
@testset "large dense lp example (dense A)" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_lp!(alf, 500, 1000, use_data=true)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ 2055.807 atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ 2055.807 atol=1e-4 rtol=1e-4
end

@testset "large sparse lp example (sparse A)" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_lp!(alf, 500, 1000, dense=false)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ Alfonso.get_dobj(alf) atol=1e-4 rtol=1e-4
end

@testset "small dense lp example (dense vs sparse A)" begin
    # dense methods
    d_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_lp!(d_alf, 50, 100, dense=true, tosparse=false)
    @time Alfonso.solve!(d_alf)
    @test Alfonso.get_status(d_alf) == :Optimal

    # sparse methods
    s_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_lp!(s_alf, 50, 100, dense=true, tosparse=true)
    @time Alfonso.solve!(s_alf)
    @test Alfonso.get_status(s_alf) == :Optimal

    @test Alfonso.get_pobj(d_alf) ≈ Alfonso.get_pobj(s_alf) atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(d_alf) ≈ Alfonso.get_dobj(s_alf) atol=1e-4 rtol=1e-4
end

@testset "1D poly envelope example (dense vs sparse A)" begin
    # dense methods
    d_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_envelope!(d_alf, 2, 5, 1, 5, use_data=true, dense=true)
    @time Alfonso.solve!(d_alf)
    @test Alfonso.get_status(d_alf) == :Optimal
    @test Alfonso.get_pobj(d_alf) ≈ -25.502777 atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(d_alf) ≈ -25.502777 atol=1e-4 rtol=1e-4

    # sparse methods
    s_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_envelope!(s_alf, 2, 5, 1, 5, use_data=true, dense=false)
    @time Alfonso.solve!(s_alf)
    @test Alfonso.get_status(s_alf) == :Optimal
    @test Alfonso.get_pobj(s_alf) ≈ -25.502777 atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(s_alf) ≈ -25.502777 atol=1e-4 rtol=1e-4
end

@testset "2D poly envelope example (dense vs sparse A)" begin
    # dense methods
    d_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_envelope!(d_alf, 2, 4, 2, 7, dense=true)
    @time Alfonso.solve!(d_alf)
    @test Alfonso.get_status(d_alf) == :Optimal

    # sparse methods
    s_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_envelope!(s_alf, 2, 4, 2, 7, dense=false)
    @time Alfonso.solve!(s_alf)
    @test Alfonso.get_status(s_alf) == :Optimal

    @test Alfonso.get_pobj(d_alf) ≈ Alfonso.get_pobj(s_alf) atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(d_alf) ≈ Alfonso.get_dobj(s_alf) atol=1e-4 rtol=1e-4
end

@testset "3D poly envelope example (sparse A)" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_envelope!(alf, 2, 3, 3, 5, dense=false)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ Alfonso.get_dobj(alf) atol=1e-4 rtol=1e-4
end

@testset "4D poly envelope example (sparse A)" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_envelope!(alf, 2, 3, 4, 4, dense=false)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ Alfonso.get_dobj(alf) atol=1e-4 rtol=1e-4
end

# most values taken from https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html
@testset "Butcher" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_namedpoly!(alf, :butcher, 2)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ -1.4393333333 atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ -1.4393333333 atol=1e-4 rtol=1e-4
end

@testset "Caprasse" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_namedpoly!(alf, :caprasse, 4)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ -3.1800966258 atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ -3.1800966258 atol=1e-4 rtol=1e-4
end

@testset "Goldstein-Price" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_namedpoly!(alf, :goldsteinprice, 7)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ 3 atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ 3 atol=1e-4 rtol=1e-4
end

# out of memory during interpolation calculations
# @testset "Heart" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_namedpoly!(alf, :heart, 2)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -1.36775 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ -1.36775 atol=1e-4 rtol=1e-4
# end

@testset "Lotka-Volterra" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_namedpoly!(alf, :lotkavolterra, 3)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ -20.8 atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ -20.8 atol=1e-4 rtol=1e-4
end

# out of memory during interpolation calculations
# @testset "Magnetism-7" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_namedpoly!(alf, :magnetism7, 2)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -0.25 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ -0.25 atol=1e-4 rtol=1e-4
# end

@testset "Motzkin" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag, optimtol=1e-5)
    build_namedpoly!(alf, :motzkin, 7)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ 0 atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ 0 atol=1e-4 rtol=1e-4
end

@testset "Reaction-diffusion" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    build_namedpoly!(alf, :reactiondiffusion, 4)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ -36.71269068 atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ -36.71269068 atol=1e-4 rtol=1e-4
end

@testset "Robinson" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag, optimtol=1e-5)
    build_namedpoly!(alf, :robinson, 8)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ 0.814814 atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ 0.814814 atol=1e-4 rtol=1e-4
end

# tolerances not satisfied
@testset "Rosenbrock" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag, optimtol=1e-5, maxpredsmallsteps=20)
    build_namedpoly!(alf, :rosenbrock, 3)
    @time Alfonso.solve!(alf)
    # @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
    @test Alfonso.get_dobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
end

# tolerances not satisfied
@testset "Schwefel" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag, optimtol=1e-5, maxpredsmallsteps=25)
    build_namedpoly!(alf, :schwefel, 4)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
    @test Alfonso.get_dobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
end

@testset "small second-order cone problem" begin
    alf = Alfonso.AlfonsoOpt(verbose=verbflag)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    cone = Alfonso.Cone([Alfonso.SecondOrderCone(3)], AbstractUnitRange[1:3])
    Alfonso.load_data!(alf, A, b, c, cone)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
    @test Alfonso.get_pobj(alf) ≈ -sqrt(2) atol=1e-4 rtol=1e-4
    @test Alfonso.get_dobj(alf) ≈ Alfonso.get_pobj(alf) atol=1e-4 rtol=1e-4
    @test Alfonso.get_y(alf) ≈ [-sqrt(2), 0.0] atol=1e-4 rtol=1e-4
    @test Alfonso.get_x(alf) ≈ [1.0, 1/sqrt(2), 1/sqrt(2)] atol=1e-4 rtol=1e-4
end
