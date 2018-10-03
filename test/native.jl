#=
Copyright 2018, Chris Coey and contributors
=#

egs_dir = joinpath(@__DIR__, "../examples")
include(joinpath(egs_dir, "envelope/envelope.jl"))
include(joinpath(egs_dir, "lp/lp.jl"))
include(joinpath(egs_dir, "namedpoly/namedpoly.jl"))

function testnative(verbflag::Bool)
    @testset "native interface tests" begin

    @testset "small lp 1: nonnegative vs nonpositive orthant" begin
        Random.seed!(1)
        (n, p, q) = (10, 8, 10)
        c = rand(0.0:9.0, n)
        A = rand(-9.0:9.0, p, n)
        b = A*ones(n)
        h = zeros(q)

        opt1 = Hypatia.Optimizer(verbose=verbflag)
        G = SparseMatrixCSC(-1.0I, q, n)
        cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q])
        Hypatia.load_data!(opt1, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt1)
        @test Hypatia.get_status(opt1) == :Optimal

        opt2 = Hypatia.Optimizer(verbose=verbflag)
        G = SparseMatrixCSC(1.0I, q, n)
        cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q])
        Hypatia.load_data!(opt2, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt2)
        @test Hypatia.get_status(opt2) == :Optimal

        @test Hypatia.get_pobj(opt1) ≈ Hypatia.get_pobj(opt2) atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt1) ≈ Hypatia.get_dobj(opt2) atol=1e-4 rtol=1e-4
    end

    @testset "small lp 2" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        Random.seed!(1)
        (n, p, q) = (5, 2, 10)
        c = rand(0.0:9.0, n)
        A = rand(-9.0:9.0, p, n)
        b = A*ones(n)
        G = rand(q, n) - Matrix(2.0I, q, n)
        h = G*ones(n)
        cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_status(opt) == :Optimal
    end

    @testset "small lp 3" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        Random.seed!(1)
        (n, p, q) = (30, 12, 30)
        c = rand(0.0:9.0, n)
        A = rand(-9.0:9.0, p, n)
        b = A*ones(n)
        @assert n == q
        G = Diagonal(1.0I, n)
        h = zeros(q)
        cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_status(opt) == :Optimal
    end

    @testset "small L_infinity cone problem" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        c = Float64[0, -1, -1]
        A = Float64[1 0 0; 0 1 0]
        b = Float64[1, 1/sqrt(2)]
        G = SparseMatrixCSC(-1.0I, 3, 3)
        h = zeros(3)
        cone = Hypatia.Cone([Hypatia.EllInfinityCone(3)], [1:3])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 20
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ -1 - 1/sqrt(2) atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ Hypatia.get_pobj(opt) atol=1e-4 rtol=1e-4
        @test Hypatia.get_y(opt) ≈ [1, 1] atol=1e-4 rtol=1e-4
        @test Hypatia.get_x(opt) ≈ [1, 1/sqrt(2), 1] atol=1e-4 rtol=1e-4
    end

    @testset "small L_infinity cone problem 2" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        Random.seed!(1)
        c = Float64[1, 0, 0, 0, 0, 0]
        A = rand(-9.0:9.0, 3, 6)
        b = A*ones(6)
        G = rand(6, 6)
        h = G*ones(6)
        cone = Hypatia.Cone([Hypatia.EllInfinityCone(6)], [1:6])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 20
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ 1 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ Hypatia.get_pobj(opt) atol=1e-4 rtol=1e-4
    end

    @testset "small second-order cone problem" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        c = Float64[0, -1, -1]
        A = Float64[1 0 0; 0 1 0]
        b = Float64[1, 1/sqrt(2)]
        G = SparseMatrixCSC(-1.0I, 3, 3)
        h = zeros(3)
        cone = Hypatia.Cone([Hypatia.SecondOrderCone(3)], [1:3])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 15
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ Hypatia.get_pobj(opt) atol=1e-4 rtol=1e-4
        @test Hypatia.get_y(opt) ≈ [sqrt(2), 0] atol=1e-4 rtol=1e-4
        @test Hypatia.get_x(opt) ≈ [1, 1/sqrt(2), 1/sqrt(2)] atol=1e-4 rtol=1e-4
    end

    @testset "small rotated second-order cone problem" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        c = Float64[0, 0, -1, -1]
        A = Float64[1 0 0 0; 0 1 0 0]
        b = Float64[1/2, 1]
        G = SparseMatrixCSC(-1.0I, 4, 4)
        h = zeros(4)
        cone = Hypatia.Cone([Hypatia.RotatedSecondOrderCone(4)], [1:4])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 15
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ Hypatia.get_pobj(opt) atol=1e-4 rtol=1e-4
        @test Hypatia.get_x(opt)[3:4] ≈ [1, 1]/sqrt(2) atol=1e-4 rtol=1e-4
    end

    @testset "small rotated second-order cone problem 2" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        c = Float64[0, 0, -1]
        A = Float64[1 0 0; 0 1 0]
        b = Float64[1/2, 1]/sqrt(2)
        G = SparseMatrixCSC(-1.0I, 3, 3)
        h = zeros(3)
        cone = Hypatia.Cone([Hypatia.RotatedSecondOrderCone(3)], [1:3])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 20
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ -1/sqrt(2) atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ Hypatia.get_pobj(opt) atol=1e-4 rtol=1e-4
        @test Hypatia.get_x(opt)[2] ≈ 1/sqrt(2) atol=1e-4 rtol=1e-4
    end

    @testset "small positive semidefinite cone problem" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        c = Float64[0, -1, 0]
        A = Float64[1 0 0; 0 0 1]
        b = Float64[1/2, 1]
        G = SparseMatrixCSC(-1.0I, 3, 3)
        h = zeros(3)
        cone = Hypatia.Cone([Hypatia.PositiveSemidefiniteCone(3)], [1:3])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 15
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ -1 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ Hypatia.get_pobj(opt) atol=1e-4 rtol=1e-4
        @test Hypatia.get_x(opt)[2] ≈ 1 atol=1e-4 rtol=1e-4
    end

    @testset "small positive semidefinite cone problem 2" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        c = Float64[1, 0, 1, 0, 0, 1]
        A = Float64[1 2 3 4 5 6; 1 1 1 1 1 1]
        b = Float64[10, 3]
        G = SparseMatrixCSC(-1.0I, 6, 6)
        h = zeros(6)
        cone = Hypatia.Cone([Hypatia.PositiveSemidefiniteCone(6)], [1:6])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 20
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ 1.249632 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ Hypatia.get_pobj(opt) atol=1e-4 rtol=1e-4
        @test Hypatia.get_x(opt) ≈ [0.491545, 0.647333, 0.426249, 0.571161, 0.531874, 0.331838] atol=1e-4 rtol=1e-4
    end

    @testset "small exponential cone problem" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        c = Float64[1, 1, 1]
        A = Float64[0 1 0; 1 0 0]
        b = Float64[2, 1]
        G = SparseMatrixCSC(-1.0I, 3, 3)
        h = zeros(3)
        cone = Hypatia.Cone([Hypatia.ExponentialCone()], [1:3])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 20
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ (2*exp(1/2)+3) atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ Hypatia.get_pobj(opt) atol=1e-4 rtol=1e-4
        @test dot(Hypatia.get_y(opt), b) ≈ -Hypatia.get_pobj(opt) atol=1e-4 rtol=1e-4
        @test Hypatia.get_x(opt) ≈ [1, 2, 2*exp(1/2)] atol=1e-4 rtol=1e-4
        @test Hypatia.get_y(opt) ≈ -[1+exp(1/2)/2, 1+exp(1/2)] atol=1e-4 rtol=1e-4
        @test Hypatia.get_z(opt) ≈ (c + A'*Hypatia.get_y(opt)) atol=1e-4 rtol=1e-4
    end

    @testset "small power cone problem" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        c = Float64[1, 0, 0, -1, -1, 0]
        A = Float64[1 1 1/2 0 0 0; 0 0 0 0 0 1]
        b = Float64[2, 1]
        G = SparseMatrixCSC(-1.0I, 6, 6)[[4, 1, 2, 5, 3, 6], :]
        h = zeros(6)
        cone = Hypatia.Cone([Hypatia.PowerCone([0.2, 0.8]), Hypatia.PowerCone([0.4, 0.6])], [1:3, 4:6])
        Hypatia.load_data!(opt, c, A, b, G, h, cone)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 25
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ -1.80734 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ Hypatia.get_pobj(opt) atol=1e-4 rtol=1e-4
        @test Hypatia.get_x(opt)[1:3] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
    end

    @testset "small dense lp example (dense vs sparse A)" begin
        # dense methods
        opt2 = Hypatia.Optimizer(verbose=verbflag)
        build_lp!(opt2, 50, 100, dense=true, tosparse=false)
        @time Hypatia.solve!(opt2)
        @test Hypatia.get_niters(opt2) <= 40
        @test Hypatia.get_status(opt2) == :Optimal

        # sparse methods
        opt1 = Hypatia.Optimizer(verbose=verbflag)
        build_lp!(opt1, 50, 100, dense=true, tosparse=true)
        @time Hypatia.solve!(opt1)
        @test Hypatia.get_niters(opt1) <= 40
        @test Hypatia.get_status(opt1) == :Optimal

        @test Hypatia.get_pobj(opt2) ≈ Hypatia.get_pobj(opt1) atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt2) ≈ Hypatia.get_dobj(opt1) atol=1e-4 rtol=1e-4
    end

    @testset "1D poly envelope example (dense vs sparse A)" begin
        # dense methods
        opt2 = Hypatia.Optimizer(verbose=verbflag)
        build_envelope!(opt2, 2, 5, 1, 5, use_data=true, dense=true)
        @time Hypatia.solve!(opt2)
        @test Hypatia.get_niters(opt2) <= 30
        @test Hypatia.get_status(opt2) == :Optimal
        @test Hypatia.get_pobj(opt2) ≈ -25.502777 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt2) ≈ -25.502777 atol=1e-4 rtol=1e-4

        # sparse methods
        opt1 = Hypatia.Optimizer(verbose=verbflag)
        build_envelope!(opt1, 2, 5, 1, 5, use_data=true, dense=false)
        @time Hypatia.solve!(opt1)
        @test Hypatia.get_niters(opt1) <= 30
        @test Hypatia.get_status(opt1) == :Optimal
        @test Hypatia.get_pobj(opt1) ≈ -25.502777 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt1) ≈ -25.502777 atol=1e-4 rtol=1e-4
    end

    # most values taken from https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html
    @testset "Butcher" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        build_namedpoly!(opt, :butcher, 2)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 40
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ -1.4393333333 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ -1.4393333333 atol=1e-4 rtol=1e-4
    end

    @testset "Caprasse" begin
        opt = Hypatia.Optimizer(verbose=verbflag, tolfeas=5e-7)
        build_namedpoly!(opt, :caprasse, 4)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 45
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ -3.1800966258 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ -3.1800966258 atol=1e-4 rtol=1e-4
    end

    @testset "Goldstein-Price" begin
        opt = Hypatia.Optimizer(verbose=verbflag, tolfeas=1e-10)
        build_namedpoly!(opt, :goldsteinprice, 7)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 60
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ 3 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ 3 atol=1e-4 rtol=1e-4
    end

    # out of memory during interpolation calculations
    # @testset "Heart" begin
    #     opt = Hypatia.Optimizer(verbose=verbflag)
    #     build_namedpoly!(opt, :heart, 2)
    #     @time Hypatia.solve!(opt)
    #     @test Hypatia.get_status(opt) == :Optimal
    #     @test Hypatia.get_pobj(opt) ≈ -1.36775 atol=1e-4 rtol=1e-4
    #     @test Hypatia.get_dobj(opt) ≈ -1.36775 atol=1e-4 rtol=1e-4
    # end

    @testset "Lotka-Volterra" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        build_namedpoly!(opt, :lotkavolterra, 3)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 35
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ -20.8 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ -20.8 atol=1e-4 rtol=1e-4
    end

    # out of memory during interpolation calculations
    # @testset "Magnetism-7" begin
    #     opt = Hypatia.Optimizer(verbose=verbflag)
    #     build_namedpoly!(opt, :magnetism7, 2)
    #     @time Hypatia.solve!(opt)
    #     @test Hypatia.get_status(opt) == :Optimal
    #     @test Hypatia.get_pobj(opt) ≈ -0.25 atol=1e-4 rtol=1e-4
    #     @test Hypatia.get_dobj(opt) ≈ -0.25 atol=1e-4 rtol=1e-4
    # end

    @testset "Motzkin" begin
        opt = Hypatia.Optimizer(verbose=verbflag, tolrelopt=1e-5, tolabsopt=1e-6, tolfeas=1e-6)
        build_namedpoly!(opt, :motzkin, 7)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 35
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ 0 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ 0 atol=1e-4 rtol=1e-4
    end

    @testset "Reaction-diffusion" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        build_namedpoly!(opt, :reactiondiffusion, 4)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 35
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ -36.71269068 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ -36.71269068 atol=1e-4 rtol=1e-4
    end

    @testset "Robinson" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        build_namedpoly!(opt, :robinson, 8)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 40
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ 0.814814 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ 0.814814 atol=1e-4 rtol=1e-4
    end

    @testset "Rosenbrock" begin
        opt = Hypatia.Optimizer(verbose=verbflag, tolfeas=1.1e-8)
        build_namedpoly!(opt, :rosenbrock, 3)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 65
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ 0 atol=1e-2 rtol=1e-2
        @test Hypatia.get_dobj(opt) ≈ 0 atol=1e-3 rtol=1e-3
    end

    @testset "Schwefel" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        build_namedpoly!(opt, :schwefel, 4)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 50
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ 0 atol=1e-3 rtol=1e-3
        @test Hypatia.get_dobj(opt) ≈ 0 atol=1e-3 rtol=1e-3
    end

    @testset "large dense lp example (dense A)" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        build_lp!(opt, 500, 1000, use_data=true, dense=true)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 75
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ 2055.807 atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt) ≈ 2055.807 atol=1e-4 rtol=1e-4
    end

    @testset "large sparse lp example (sparse A)" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        build_lp!(opt, 500, 1000, dense=false, nzfrac=10/1000)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_niters(opt) <= 70
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ Hypatia.get_dobj(opt) atol=1e-4 rtol=1e-4
    end

    @testset "2D poly envelope example (dense vs sparse A)" begin
        # dense methods
        opt2 = Hypatia.Optimizer(verbose=verbflag)
        build_envelope!(opt2, 2, 4, 2, 7, dense=true)
        @time Hypatia.solve!(opt2)
        @test Hypatia.get_niters(opt2) <= 55
        @test Hypatia.get_status(opt2) == :Optimal

        # sparse methods
        opt1 = Hypatia.Optimizer(verbose=verbflag)
        build_envelope!(opt1, 2, 4, 2, 7, dense=false)
        @time Hypatia.solve!(opt1)
        @test Hypatia.get_niters(opt1) <= 55
        @test Hypatia.get_status(opt1) == :Optimal

        @test Hypatia.get_pobj(opt2) ≈ Hypatia.get_pobj(opt1) atol=1e-4 rtol=1e-4
        @test Hypatia.get_dobj(opt2) ≈ Hypatia.get_dobj(opt1) atol=1e-4 rtol=1e-4
    end

    @testset "3D poly envelope example (sparse A)" begin
        opt = Hypatia.Optimizer(verbose=verbflag)
        build_envelope!(opt, 2, 3, 3, 5, dense=false)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ Hypatia.get_dobj(opt) atol=1e-4 rtol=1e-4
    end

    @testset "4D poly envelope example (sparse A)" begin
        opt = Hypatia.Optimizer(verbose=verbflag, tolrelopt=1e-5, tolabsopt=1e-6, tolfeas=1e-6)
        build_envelope!(opt, 2, 3, 4, 4, dense=false)
        @time Hypatia.solve!(opt)
        @test Hypatia.get_status(opt) == :Optimal
        @test Hypatia.get_pobj(opt) ≈ Hypatia.get_dobj(opt) atol=1e-4 rtol=1e-4
    end

    end
    return nothing
end
