#=
Copyright 2018, Chris Coey and contributors
=#

@testset "small lp 1: nonnegative vs nonpositive orthant" begin
    Random.seed!(1)
    (n, p, q) = (10, 8, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    h = zeros(q)

    alf1 = Hypatia.HypatiaOpt(verbose=verbflag)
    G = SparseMatrixCSC(-1.0I, q, n)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q])
    Hypatia.load_data!(alf1, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf1)
    @test Hypatia.get_status(alf1) == :Optimal

    alf2 = Hypatia.HypatiaOpt(verbose=verbflag)
    G = SparseMatrixCSC(1.0I, q, n)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q])
    Hypatia.load_data!(alf2, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf2)
    @test Hypatia.get_status(alf2) == :Optimal

    @test Hypatia.get_pobj(alf1) ≈ Hypatia.get_pobj(alf2) atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf1) ≈ Hypatia.get_dobj(alf2) atol=1e-4 rtol=1e-4
end

@testset "small lp 2" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G*ones(n)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_status(alf) == :Optimal
end

@testset "small lp 3" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    Random.seed!(1)
    (n, p, q) = (30, 12, 30)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    @assert n == q
    G = Diagonal(1.0I, n)
    h = zeros(q)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_status(alf) == :Optimal
end

@testset "small L_infinity cone problem" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.EllInfinityCone(3)], [1:3])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 20
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ -1 - 1/sqrt(2) atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ Hypatia.get_pobj(alf) atol=1e-4 rtol=1e-4
    @test Hypatia.get_y(alf) ≈ [1, 1] atol=1e-4 rtol=1e-4
    @test Hypatia.get_x(alf) ≈ [1, 1/sqrt(2), 1] atol=1e-4 rtol=1e-4
end

@testset "small L_infinity cone problem 2" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A*ones(6)
    G = rand(6, 6)
    h = G*ones(6)
    cone = Hypatia.Cone([Hypatia.EllInfinityCone(6)], [1:6])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 20
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ 1 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ Hypatia.get_pobj(alf) atol=1e-4 rtol=1e-4
end

@testset "small second-order cone problem" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.SecondOrderCone(3)], [1:3])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 15
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ -sqrt(2) atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ Hypatia.get_pobj(alf) atol=1e-4 rtol=1e-4
    @test Hypatia.get_y(alf) ≈ [sqrt(2), 0] atol=1e-4 rtol=1e-4
    @test Hypatia.get_x(alf) ≈ [1, 1/sqrt(2), 1/sqrt(2)] atol=1e-4 rtol=1e-4
end

@testset "small rotated second-order cone problem" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    c = Float64[0, 0, -1, -1]
    A = Float64[1 0 0 0; 0 1 0 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)
    cone = Hypatia.Cone([Hypatia.RotatedSecondOrderCone(4)], [1:4])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 15
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ -sqrt(2) atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ Hypatia.get_pobj(alf) atol=1e-4 rtol=1e-4
    @test Hypatia.get_x(alf)[3:4] ≈ [1, 1]/sqrt(2) atol=1e-4 rtol=1e-4
end

@testset "small rotated second-order cone problem 2" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    c = Float64[0, 0, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1/2, 1]/sqrt(2)
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.RotatedSecondOrderCone(3)], [1:3])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 20
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ -1/sqrt(2) atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ Hypatia.get_pobj(alf) atol=1e-4 rtol=1e-4
    @test Hypatia.get_x(alf)[2] ≈ 1/sqrt(2) atol=1e-4 rtol=1e-4
end

@testset "small positive semidefinite cone problem" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    c = Float64[0, -1, 0]
    A = Float64[1 0 0; 0 0 1]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.PositiveSemidefiniteCone(3)], [1:3])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 15
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ -1 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ Hypatia.get_pobj(alf) atol=1e-4 rtol=1e-4
    @test Hypatia.get_x(alf)[2] ≈ 1 atol=1e-4 rtol=1e-4
end

@testset "small positive semidefinite cone problem 2" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    c = Float64[1, 0, 1, 0, 0, 1]
    A = Float64[1 2 3 4 5 6; 1 1 1 1 1 1]
    b = Float64[10, 3]
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)
    cone = Hypatia.Cone([Hypatia.PositiveSemidefiniteCone(6)], [1:6])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 20
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ 1.249632 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ Hypatia.get_pobj(alf) atol=1e-4 rtol=1e-4
    @test Hypatia.get_x(alf) ≈ [0.491545, 0.647333, 0.426249, 0.571161, 0.531874, 0.331838] atol=1e-4 rtol=1e-4
end

@testset "small exponential cone problem" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    c = Float64[1, 1, 1]
    A = Float64[0 1 0; 1 0 0]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.ExponentialCone()], [1:3])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 20
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ (2*exp(1/2)+3) atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ Hypatia.get_pobj(alf) atol=1e-4 rtol=1e-4
    @test dot(Hypatia.get_y(alf), b) ≈ -Hypatia.get_pobj(alf) atol=1e-4 rtol=1e-4
    @test Hypatia.get_x(alf) ≈ [1, 2, 2*exp(1/2)] atol=1e-4 rtol=1e-4
    @test Hypatia.get_y(alf) ≈ -[1+exp(1/2)/2, 1+exp(1/2)] atol=1e-4 rtol=1e-4
    @test Hypatia.get_z(alf) ≈ (c + A'*Hypatia.get_y(alf)) atol=1e-4 rtol=1e-4
end

@testset "small power cone problem" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    c = Float64[1, 0, 0, -1, -1, 0]
    A = Float64[1 1 1/2 0 0 0; 0 0 0 0 0 1]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 6, 6)[[4, 1, 2, 5, 3, 6], :]
    h = zeros(6)
    cone = Hypatia.Cone([Hypatia.PowerCone([0.2, 0.8]), Hypatia.PowerCone([0.4, 0.6])], [1:3, 4:6])
    Hypatia.load_data!(alf, c, A, b, G, h, cone)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 25
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ -1.80734 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ Hypatia.get_pobj(alf) atol=1e-4 rtol=1e-4
    @test Hypatia.get_x(alf)[1:3] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
end

@testset "small dense lp example (dense vs sparse A)" begin
    # dense methods
    alf2 = Hypatia.HypatiaOpt(verbose=verbflag)
    build_lp!(alf2, 50, 100, dense=true, tosparse=false)
    @time Hypatia.solve!(alf2)
    @test Hypatia.get_niters(alf2) <= 40
    @test Hypatia.get_status(alf2) == :Optimal

    # sparse methods
    alf1 = Hypatia.HypatiaOpt(verbose=verbflag)
    build_lp!(alf1, 50, 100, dense=true, tosparse=true)
    @time Hypatia.solve!(alf1)
    @test Hypatia.get_niters(alf1) <= 40
    @test Hypatia.get_status(alf1) == :Optimal

    @test Hypatia.get_pobj(alf2) ≈ Hypatia.get_pobj(alf1) atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf2) ≈ Hypatia.get_dobj(alf1) atol=1e-4 rtol=1e-4
end

@testset "1D poly envelope example (dense vs sparse A)" begin
    # dense methods
    alf2 = Hypatia.HypatiaOpt(verbose=verbflag)
    build_envelope!(alf2, 2, 5, 1, 5, use_data=true, dense=true)
    @time Hypatia.solve!(alf2)
    @test Hypatia.get_niters(alf2) <= 30
    @test Hypatia.get_status(alf2) == :Optimal
    @test Hypatia.get_pobj(alf2) ≈ -25.502777 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf2) ≈ -25.502777 atol=1e-4 rtol=1e-4

    # sparse methods
    alf1 = Hypatia.HypatiaOpt(verbose=verbflag)
    build_envelope!(alf1, 2, 5, 1, 5, use_data=true, dense=false)
    @time Hypatia.solve!(alf1)
    @test Hypatia.get_niters(alf1) <= 30
    @test Hypatia.get_status(alf1) == :Optimal
    @test Hypatia.get_pobj(alf1) ≈ -25.502777 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf1) ≈ -25.502777 atol=1e-4 rtol=1e-4
end

# most values taken from https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html
@testset "Butcher" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    build_namedpoly!(alf, :butcher, 2)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 40
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ -1.4393333333 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ -1.4393333333 atol=1e-4 rtol=1e-4
end

@testset "Caprasse" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag, tolfeas=5e-7)
    build_namedpoly!(alf, :caprasse, 4)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 45
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ -3.1800966258 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ -3.1800966258 atol=1e-4 rtol=1e-4
end

@testset "Goldstein-Price" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag, tolfeas=1e-10)
    build_namedpoly!(alf, :goldsteinprice, 7)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 60
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ 3 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ 3 atol=1e-4 rtol=1e-4
end

# out of memory during interpolation calculations
# @testset "Heart" begin
#     alf = Hypatia.HypatiaOpt(verbose=verbflag)
#     build_namedpoly!(alf, :heart, 2)
#     @time Hypatia.solve!(alf)
#     @test Hypatia.get_status(alf) == :Optimal
#     @test Hypatia.get_pobj(alf) ≈ -1.36775 atol=1e-4 rtol=1e-4
#     @test Hypatia.get_dobj(alf) ≈ -1.36775 atol=1e-4 rtol=1e-4
# end

@testset "Lotka-Volterra" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    build_namedpoly!(alf, :lotkavolterra, 3)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 35
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ -20.8 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ -20.8 atol=1e-4 rtol=1e-4
end

# out of memory during interpolation calculations
# @testset "Magnetism-7" begin
#     alf = Hypatia.HypatiaOpt(verbose=verbflag)
#     build_namedpoly!(alf, :magnetism7, 2)
#     @time Hypatia.solve!(alf)
#     @test Hypatia.get_status(alf) == :Optimal
#     @test Hypatia.get_pobj(alf) ≈ -0.25 atol=1e-4 rtol=1e-4
#     @test Hypatia.get_dobj(alf) ≈ -0.25 atol=1e-4 rtol=1e-4
# end

@testset "Motzkin" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag, tolrelopt=1e-5, tolabsopt=1e-6, tolfeas=1e-6)
    build_namedpoly!(alf, :motzkin, 7)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 35
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ 0 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ 0 atol=1e-4 rtol=1e-4
end

@testset "Reaction-diffusion" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    build_namedpoly!(alf, :reactiondiffusion, 4)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 35
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ -36.71269068 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ -36.71269068 atol=1e-4 rtol=1e-4
end

@testset "Robinson" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    build_namedpoly!(alf, :robinson, 8)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 40
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ 0.814814 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ 0.814814 atol=1e-4 rtol=1e-4
end

@testset "Rosenbrock" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag, tolfeas=1.1e-8)
    build_namedpoly!(alf, :rosenbrock, 3)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 65
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ 0 atol=1e-2 rtol=1e-2
    @test Hypatia.get_dobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
end

@testset "Schwefel" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    build_namedpoly!(alf, :schwefel, 4)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 50
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
    @test Hypatia.get_dobj(alf) ≈ 0 atol=1e-3 rtol=1e-3
end

@testset "large dense lp example (dense A)" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    build_lp!(alf, 500, 1000, use_data=true, dense=true)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 75
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ 2055.807 atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf) ≈ 2055.807 atol=1e-4 rtol=1e-4
end

@testset "large sparse lp example (sparse A)" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    build_lp!(alf, 500, 1000, dense=false, nzfrac=10/1000)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_niters(alf) <= 70
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ Hypatia.get_dobj(alf) atol=1e-4 rtol=1e-4
end

@testset "2D poly envelope example (dense vs sparse A)" begin
    # dense methods
    alf2 = Hypatia.HypatiaOpt(verbose=verbflag)
    build_envelope!(alf2, 2, 4, 2, 7, dense=true)
    @time Hypatia.solve!(alf2)
    @test Hypatia.get_niters(alf2) <= 55
    @test Hypatia.get_status(alf2) == :Optimal

    # sparse methods
    alf1 = Hypatia.HypatiaOpt(verbose=verbflag)
    build_envelope!(alf1, 2, 4, 2, 7, dense=false)
    @time Hypatia.solve!(alf1)
    @test Hypatia.get_niters(alf1) <= 55
    @test Hypatia.get_status(alf1) == :Optimal

    @test Hypatia.get_pobj(alf2) ≈ Hypatia.get_pobj(alf1) atol=1e-4 rtol=1e-4
    @test Hypatia.get_dobj(alf2) ≈ Hypatia.get_dobj(alf1) atol=1e-4 rtol=1e-4
end

@testset "3D poly envelope example (sparse A)" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag)
    build_envelope!(alf, 2, 3, 3, 5, dense=false)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ Hypatia.get_dobj(alf) atol=1e-4 rtol=1e-4
end

@testset "4D poly envelope example (sparse A)" begin
    alf = Hypatia.HypatiaOpt(verbose=verbflag, tolrelopt=1e-5, tolabsopt=1e-6, tolfeas=1e-6)
    build_envelope!(alf, 2, 3, 4, 4, dense=false)
    @time Hypatia.solve!(alf)
    @test Hypatia.get_status(alf) == :Optimal
    @test Hypatia.get_pobj(alf) ≈ Hypatia.get_dobj(alf) atol=1e-4 rtol=1e-4
end
