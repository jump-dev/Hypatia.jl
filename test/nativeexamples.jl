#
# @testset "large dense lp example (dense A)" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_lp!(alf, 500, 1000, use_data=true, dense=true)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ 2055.807 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ 2055.807 atol=1e-4 rtol=1e-4
# end
#
# @testset "large sparse lp example (sparse A)" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_lp!(alf, 500, 1000, dense=false)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ Alfonso.get_dobj(alf) atol=1e-4 rtol=1e-4
# end

# @testset "small dense lp example (dense vs sparse A)" begin
#     # dense methods
#     d_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_lp!(d_alf, 50, 100, dense=true, tosparse=false)
#     @time Alfonso.solve!(d_alf)
#     @test Alfonso.get_status(d_alf) == :Optimal
#
#     # sparse methods
#     s_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_lp!(s_alf, 50, 100, dense=true, tosparse=true)
#     @time Alfonso.solve!(s_alf)
#     @test Alfonso.get_status(s_alf) == :Optimal
#
#     @test Alfonso.get_pobj(d_alf) ≈ Alfonso.get_pobj(s_alf) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(d_alf) ≈ Alfonso.get_dobj(s_alf) atol=1e-4 rtol=1e-4
# end

# @testset "1D poly envelope example (dense vs sparse A)" begin
#     # dense methods
#     d_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_envelope!(d_alf, 2, 5, 1, 5, use_data=true, dense=true)
#     @time Alfonso.solve!(d_alf)
#     @test Alfonso.get_status(d_alf) == :Optimal
#     @test Alfonso.get_pobj(d_alf) ≈ -25.502777 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(d_alf) ≈ -25.502777 atol=1e-4 rtol=1e-4
#
#     # sparse methods
#     s_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_envelope!(s_alf, 2, 5, 1, 5, use_data=true, dense=false)
#     @time Alfonso.solve!(s_alf)
#     @test Alfonso.get_status(s_alf) == :Optimal
#     @test Alfonso.get_pobj(s_alf) ≈ -25.502777 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(s_alf) ≈ -25.502777 atol=1e-4 rtol=1e-4
# end

# @testset "2D poly envelope example (dense vs sparse A)" begin
#     # dense methods
#     d_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_envelope!(d_alf, 2, 4, 2, 7, dense=true)
#     @time Alfonso.solve!(d_alf)
#     @test Alfonso.get_status(d_alf) == :Optimal
#
#     # sparse methods
#     s_alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_envelope!(s_alf, 2, 4, 2, 7, dense=false)
#     @time Alfonso.solve!(s_alf)
#     @test Alfonso.get_status(s_alf) == :Optimal
#
#     @test Alfonso.get_pobj(d_alf) ≈ Alfonso.get_pobj(s_alf) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(d_alf) ≈ Alfonso.get_dobj(s_alf) atol=1e-4 rtol=1e-4
# end
#
# @testset "3D poly envelope example (sparse A)" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_envelope!(alf, 2, 3, 3, 5, dense=false)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ Alfonso.get_dobj(alf) atol=1e-4 rtol=1e-4
# end
#
# # @testset "4D poly envelope example (sparse A)" begin
# #     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
# #     build_envelope!(alf, 2, 3, 4, 4, dense=false)
# #     @time Alfonso.solve!(alf)
# #     @test Alfonso.get_status(alf) == :Optimal
# #     @test Alfonso.get_pobj(alf) ≈ Alfonso.get_dobj(alf) atol=1e-4 rtol=1e-4
# # end
#
# most values taken from https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html
# @testset "Butcher" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_namedpoly!(alf, :butcher, 2)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -1.4393333333 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ -1.4393333333 atol=1e-4 rtol=1e-4
# end
#
# @testset "Caprasse" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_namedpoly!(alf, :caprasse, 4)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -3.1800966258 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ -3.1800966258 atol=1e-4 rtol=1e-4
# end

# # @testset "Goldstein-Price" begin
# #     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
# #     build_namedpoly!(alf, :goldsteinprice, 7)
# #     @time Alfonso.solve!(alf)
# #     @test Alfonso.get_status(alf) == :Optimal
# #     @test Alfonso.get_pobj(alf) ≈ 3 atol=1e-4 rtol=1e-4
# #     @test Alfonso.get_dobj(alf) ≈ 3 atol=1e-4 rtol=1e-4
# # end
#
# # out of memory during interpolation calculations
# # @testset "Heart" begin
# #     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
# #     build_namedpoly!(alf, :heart, 2)
# #     @time Alfonso.solve!(alf)
# #     @test Alfonso.get_status(alf) == :Optimal
# #     @test Alfonso.get_pobj(alf) ≈ -1.36775 atol=1e-4 rtol=1e-4
# #     @test Alfonso.get_dobj(alf) ≈ -1.36775 atol=1e-4 rtol=1e-4
# # end
#
# @testset "Lotka-Volterra" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_namedpoly!(alf, :lotkavolterra, 3)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -20.8 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ -20.8 atol=1e-4 rtol=1e-4
# end
#
# # out of memory during interpolation calculations
# # @testset "Magnetism-7" begin
# #     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
# #     build_namedpoly!(alf, :magnetism7, 2)
# #     @time Alfonso.solve!(alf)
# #     @test Alfonso.get_status(alf) == :Optimal
# #     @test Alfonso.get_pobj(alf) ≈ -0.25 atol=1e-4 rtol=1e-4
# #     @test Alfonso.get_dobj(alf) ≈ -0.25 atol=1e-4 rtol=1e-4
# # end
#
# @testset "Motzkin" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag, tolrelopt=1e-5, tolabsopt=1e-6, tolfeas=1e-6)
#     build_namedpoly!(alf, :motzkin, 7)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ 0 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ 0 atol=1e-4 rtol=1e-4
# end
#
# @testset "Reaction-diffusion" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     build_namedpoly!(alf, :reactiondiffusion, 4)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -36.71269068 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ -36.71269068 atol=1e-4 rtol=1e-4
# end
#
# @testset "Robinson" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag, tolrelopt=1e-5, tolabsopt=1e-6, tolfeas=1e-6)
#     build_namedpoly!(alf, :robinson, 8)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ 0.814814 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ 0.814814 atol=1e-4 rtol=1e-4
# end

# # tolerances not satisfied
# @testset "Rosenbrock" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag, tolrelopt=1e-5, tolabsopt=1e-6, tolfeas=1e-6, maxpredsmallsteps=20)
#     build_namedpoly!(alf, :rosenbrock, 3)
#     @time Alfonso.solve!(alf)
#     # @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
#     @test Alfonso.get_dobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
# end
#
# # tolerances not satisfied
# @testset "Schwefel" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag, tolrelopt=1e-5, tolabsopt=1e-6, tolfeas=1e-6, maxpredsmallsteps=25)
#     build_namedpoly!(alf, :schwefel, 4)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
#     @test Alfonso.get_dobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
# end

# @testset "small second-order cone problem" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     c = Float64[0, -1, -1]
#     A = Float64[1 0 0; 0 1 0]
#     b = Float64[1, 1/sqrt(2)]
#     cone = Alfonso.Cone([Alfonso.SecondOrderCone(3)], [1:3])
#     Alfonso.load_data!(alf, A, b, c, cone)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -sqrt(2) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ Alfonso.get_pobj(alf) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_y(alf) ≈ [-sqrt(2), 0] atol=1e-4 rtol=1e-4
#     @test Alfonso.get_x(alf) ≈ [1, 1/sqrt(2), 1/sqrt(2)] atol=1e-4 rtol=1e-4
# end
#
# @testset "small exponential cone problem" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     c = Float64[1, 1, 1]
#     A = Float64[0 1 0; 1 0 0]
#     b = Float64[2, 1]
#     cone = Alfonso.Cone([Alfonso.ExponentialCone()], [1:3])
#     Alfonso.load_data!(alf, A, b, c, cone)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ (2*exp(1/2)+3) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ Alfonso.get_pobj(alf) atol=1e-4 rtol=1e-4
#     @test dot(Alfonso.get_y(alf), b) ≈ Alfonso.get_pobj(alf) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_x(alf) ≈ [1, 2, 2*exp(1/2)] atol=1e-4 rtol=1e-4
#     @test Alfonso.get_y(alf) ≈ [1+exp(1/2)/2, 1+exp(1/2)] atol=1e-4 rtol=1e-4
#     @test Alfonso.get_s(alf) ≈ (c - A'*Alfonso.get_y(alf)) atol=1e-4 rtol=1e-4
# end
#
# @testset "small power cone problem" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     c = Float64[1, 0, 0, -1, -1, 0]
#     A = Float64[1 1 1/2 0 0 0; 0 0 0 0 0 1]
#     b = Float64[2, 1]
#     cone = Alfonso.Cone([Alfonso.PowerCone(0.2), Alfonso.PowerCone(0.4)], [[1, 2, 4], [3, 6, 5]])
#     Alfonso.load_data!(alf, A, b, c, cone)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -1.80734 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ Alfonso.get_pobj(alf) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_x(alf)[1:3] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
# end
#
# @testset "small rotated second-order cone problem" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     c = Float64[0, 0, -1, -1]
#     A = Float64[1 0 0 0; 0 1 0 0]
#     b = Float64[1/2, 1]
#     cone = Alfonso.Cone([Alfonso.RotatedSecondOrderCone(4)], [1:4])
#     Alfonso.load_data!(alf, A, b, c, cone)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -sqrt(2) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ Alfonso.get_pobj(alf) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_x(alf)[3:4] ≈ [1, 1]/sqrt(2) atol=1e-4 rtol=1e-4
# end
#
# @testset "small rotated second-order cone problem 2" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     c = Float64[0, 0, -1]
#     A = Float64[1 0 0; 0 1 0]
#     b = Float64[1/2, 1]/sqrt(2)
#     cone = Alfonso.Cone([Alfonso.RotatedSecondOrderCone(3)], [1:3])
#     Alfonso.load_data!(alf, A, b, c, cone)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -1/sqrt(2) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ Alfonso.get_pobj(alf) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_x(alf)[2] ≈ 1/sqrt(2) atol=1e-4 rtol=1e-4
# end
#
# @testset "small positive semidefinite cone problem" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     c = Float64[0, -1, 0]
#     A = Float64[1 0 0; 0 0 1]
#     b = Float64[1/2, 1]
#     cone = Alfonso.Cone([Alfonso.PositiveSemidefiniteCone(3)], [1:3])
#     Alfonso.load_data!(alf, A, b, c, cone)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ -1 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ Alfonso.get_pobj(alf) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_x(alf)[2] ≈ 1 atol=1e-4 rtol=1e-4
# end
#
# @testset "small positive semidefinite cone problem 2" begin
#     alf = Alfonso.AlfonsoOpt(verbose=verbflag)
#     c = Float64[1, 0, 1, 0, 0, 1]
#     A = Float64[1 2 3 4 5 6; 1 1 1 1 1 1]
#     b = Float64[10, 3]
#     cone = Alfonso.Cone([Alfonso.PositiveSemidefiniteCone(6)], [1:6])
#     Alfonso.load_data!(alf, A, b, c, cone)
#     @time Alfonso.solve!(alf)
#     @test Alfonso.get_status(alf) == :Optimal
#     @test Alfonso.get_pobj(alf) ≈ 1.249632 atol=1e-4 rtol=1e-4
#     @test Alfonso.get_dobj(alf) ≈ Alfonso.get_pobj(alf) atol=1e-4 rtol=1e-4
#     @test Alfonso.get_x(alf) ≈ [0.491545, 0.647333, 0.426249, 0.571161, 0.531874, 0.331838] atol=1e-4 rtol=1e-4
# end


@testset "small LP 1" begin
    (n, p, q) = (30, 12, 30)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = Matrix{Float64}(-1.0I, q, n)
    h = zeros(q)
    cone = Alfonso.Cone([Alfonso.NonnegativeCone(q)], [1:q])
    alf = Alfonso.AlfonsoOpt(verbose=true)
    Alfonso.load_data!(alf, c, A, b, G, h, cone)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
end


@testset "small LP 2" begin
    (n, p, q) = (10, 8, 9)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = Matrix{Float64}(-1.0I, q, n)
    h = zeros(q)
    cone = Alfonso.Cone([Alfonso.NonnegativeCone(q)], [1:q])
    alf = Alfonso.AlfonsoOpt(verbose=true)
    Alfonso.load_data!(alf, c, A, b, G, h, cone)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) == :Optimal
end


@testset "small LP 3" begin
    (n, p, q) = (5, 2, 5)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = Matrix{Float64}(1.0I, q, n)
    h = G*ones(n)
    cone = Alfonso.Cone([Alfonso.NonnegativeCone(q)], [1:q])
    alf = Alfonso.AlfonsoOpt(verbose=true)
    Alfonso.load_data!(alf, c, A, b, G, h, cone)
    @time Alfonso.solve!(alf)
    @test Alfonso.get_status(alf) in (:DualInfeasible, :Optimal)
end
