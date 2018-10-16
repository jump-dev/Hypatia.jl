#=
Copyright 2018, Chris Coey and contributors
=#

function _dimension1(verbose::Bool, lscachetype)
    A = Matrix{Float64}(undef, 0, 2)
    b = Float64[]
    G = [1.0 0.0]
    h = [1.0]
    c = [-1.0, 0.0]
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(1)], [1:1], [false])

    opt = Hypatia.Optimizer(verbose=verbose)
    r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    r.status == :Optimal
    @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 0] atol=1e-4 rtol=1e-4
    @test isempty(r.y)

    c = [-1.0, -1.0]
    Hypatia.check_data(c, A, b, G, h, cone)
    @test_throws ErrorException("some dual equality constraints are inconsistent") Hypatia.preprocess_data(c, A, b, G, useQR=true)

    A = sparse(A)
    G = sparse(G)
    c = [-1.0, 0.0]
    opt = Hypatia.Optimizer(verbose=verbose)
    r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    r.status == :Optimal
    @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 0] atol=1e-4 rtol=1e-4
    @test isempty(r.y)

    c = [-1.0, -1.0]
    Hypatia.check_data(c, A, b, G, h, cone)
    @test_throws ErrorException("some dual equality constraints are inconsistent") Hypatia.preprocess_data(c, A, b, G, useQR=true)
end

function _consistent1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    A = rand(-9.0:9.0, p, n)
    G = Matrix(1.0I, q, n)
    c = rand(0.0:9.0, n)
    rnd1 = rand()
    rnd2 = rand()
    A[11:15,:] = rnd1*A[1:5,:] - rnd2*A[6:10,:]
    b = A*ones(n)
    rnd1 = rand()
    rnd2 = rand()
    A[:,11:15] = rnd1*A[:,1:5] - rnd2*A[:,6:10]
    G[:,11:15] = rnd1*G[:,1:5] - rnd2*G[:,6:10]
    c[11:15] = rnd1*c[1:5] - rnd2*c[6:10]
    h = zeros(q)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q])
    r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
end

function _inconsistent1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    A = rand(-9.0:9.0, p, n)
    G = Matrix(-1.0I, q, n)
    c = rand(0.0:9.0, n)
    b = rand(p)
    rnd1 = rand()
    rnd2 = rand()
    A[11:15,:] = rnd1*A[1:5,:] - rnd2*A[6:10,:]
    b[11:15] = 2*(rnd1*b[1:5] - rnd2*b[6:10])
    h = zeros(q)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q])
    @test_throws ErrorException("some primal equality constraints are inconsistent") solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
end

function _inconsistent2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    Random.seed!(1)
    (n, p, q) = (30, 15, 30)
    A = rand(-9.0:9.0, p, n)
    G = Matrix(-1.0I, q, n)
    c = rand(0.0:9.0, n)
    b = rand(p)
    rnd1 = rand()
    rnd2 = rand()
    A[:,11:15] = rnd1*A[:,1:5] - rnd2*A[:,6:10]
    G[:,11:15] = rnd1*G[:,1:5] - rnd2*G[:,6:10]
    c[11:15] = 2*(rnd1*c[1:5] - rnd2*c[6:10])
    h = zeros(q)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q])
    @test_throws ErrorException("some dual equality constraints are inconsistent") solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
end

function _orthant1(verbose::Bool, lscachetype)
    Random.seed!(1)
    (n, p, q) = (40, 20, 40)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    h = zeros(q)

    # nonnegative cone
    opt = Hypatia.Optimizer(verbose=verbose)
    G = SparseMatrixCSC(-1.0I, q, n)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q])
    rnn = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test rnn.status == :Optimal

    # nonpositive cone
    opt = Hypatia.Optimizer(verbose=verbose)
    G = SparseMatrixCSC(1.0I, q, n)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q])
    rnp = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test rnp.status == :Optimal

    @test rnp.pobj ≈ rnn.pobj atol=1e-4 rtol=1e-4
end

function _orthant2(verbose::Bool, lscachetype)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G*ones(n)

    opt = Hypatia.Optimizer(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q], [true])
    r1 = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r1.status == :Optimal

    opt = Hypatia.Optimizer(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q], [false])
    r2 = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r2.status == :Optimal

    @test r1.pobj ≈ r2.pobj atol=1e-4 rtol=1e-4
end

function _orthant3(verbose::Bool, lscachetype)
    Random.seed!(1)
    (n, p, q) = (30, 12, 30)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = Diagonal(1.0I, n)
    h = zeros(q)

    opt = Hypatia.Optimizer(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q], [true])
    r1 = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r1.status == :Optimal

    opt = Hypatia.Optimizer(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q], [false])
    r2 = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r2.status == :Optimal

    @test r1.pobj ≈ r2.pobj atol=1e-4 rtol=1e-4
end

function _orthant4(verbose::Bool, lscachetype)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G*ones(n)

    opt = Hypatia.Optimizer(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(4), Hypatia.NonnegativeCone(6)], [1:4, 5:10], [false, true])
    r1 = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r1.status == :Optimal

    opt = Hypatia.Optimizer(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(10)], [1:10], [false])
    r2 = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r2.status == :Optimal

    @test r1.pobj ≈ r2.pobj atol=1e-4 rtol=1e-4
end

function _ellinf1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.EllInfinityCone(3)], [1:3])
    r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ -1 - 1/sqrt(2) atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 1/sqrt(2), 1] atol=1e-4 rtol=1e-4
    @test r.y ≈ [1, 1] atol=1e-4 rtol=1e-4
end

function _ellinf2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A*ones(6)
    G = rand(6, 6)
    h = G*ones(6)
    cone = Hypatia.Cone([Hypatia.EllInfinityCone(6)], [1:6])
    r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.pobj ≈ 1 atol=1e-4 rtol=1e-4
end

function _ellinfdual1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[0, 1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, -0.4]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.EllInfinityCone(3)], [1:3], [true])
    r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, -0.4, 0.6] atol=1e-4 rtol=1e-4
    @test r.y ≈ [1, 0] atol=1e-4 rtol=1e-4
end

function _ellinfdual2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A*ones(6)
    G = rand(6, 6)
    h = G*ones(6)
    cone = Hypatia.Cone([Hypatia.EllInfinityCone(6)], [1:6], [true])
    r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ 1 atol=1e-4 rtol=1e-4
end

function _ellinfdual3(verbose::Bool, lscachetype)
    Random.seed!(1)
    n = 15
    c = collect(-7.0:7.0)
    A = spzeros(2, n)
    A[1,1] = A[1,n] = A[2,1] = 1.0; A[2,n] = -1.0
    b = [0.0, 0.0]
    G = [spzeros(1, n); sparse(1.0I, n, n); spzeros(1, n); sparse(2.0I, n, n)]
    h = zeros(2n+2); h[1] = 1.0; h[n+2] = 1.0
    opt = Hypatia.Optimizer(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.EllInfinityCone(n+1), Hypatia.EllInfinityCone(n+1)], [1:n+1, n+2:2n+2], [true, false])
    r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.pobj ≈ -6.0 atol=1e-4 rtol=1e-4
    @test r.x[2] ≈ 0.5 atol=1e-4 rtol=1e-4
    @test r.x[14] ≈ -0.5 atol=1e-4 rtol=1e-4
    @test sum(abs, r.x) ≈ 1.0 atol=1e-4 rtol=1e-4
end

function _soc1(verbose::Bool, lscachetype)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for usedual in [true, false]
        opt = Hypatia.Optimizer(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.SecondOrderCone(3)], [1:3], [usedual])
        r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x ≈ [1, 1/sqrt(2), 1/sqrt(2)] atol=1e-4 rtol=1e-4
        @test r.y ≈ [sqrt(2), 0] atol=1e-4 rtol=1e-4
    end
end

function _rsoc1(verbose::Bool, lscachetype)
    c = Float64[0, 0, -1, -1]
    A = Float64[1 0 0 0; 0 1 0 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)

    for usedual in [true, false]
        opt = Hypatia.Optimizer(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.RotatedSecondOrderCone(4)], [1:4], [usedual])
        r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x[3:4] ≈ [1, 1]/sqrt(2) atol=1e-4 rtol=1e-4
    end
end

function _rsoc2(verbose::Bool, lscachetype)
    c = Float64[0, 0, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1/2, 1]/sqrt(2)
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for usedual in [true, false]
        opt = Hypatia.Optimizer(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.RotatedSecondOrderCone(3)], [1:3], [usedual])
        r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ -1/sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x[2] ≈ 1/sqrt(2) atol=1e-4 rtol=1e-4
    end
end

function _psd1(verbose::Bool, lscachetype)
    c = Float64[0, -1, 0]
    A = Float64[1 0 0; 0 0 1]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for usedual in [true, false]
        opt = Hypatia.Optimizer(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.PositiveSemidefiniteCone(3)], [1:3], [usedual])
        r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
        @test r.x[2] ≈ 1 atol=1e-4 rtol=1e-4
    end
end

function _psd2(verbose::Bool, lscachetype)
    c = Float64[1, 0, 1, 0, 0, 1]
    A = Float64[1 2 3 4 5 6; 1 1 1 1 1 1]
    b = Float64[10, 3]
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)

    for usedual in [true, false]
        opt = Hypatia.Optimizer(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.PositiveSemidefiniteCone(6)], [1:6], [usedual])
        r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ 1.249632 atol=1e-4 rtol=1e-4
        @test r.x ≈ [0.491545, 0.647333, 0.426249, 0.571161, 0.531874, 0.331838] atol=1e-4 rtol=1e-4
    end
end

function _exp1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[1, 1, 1]
    A = Float64[0 1 0; 1 0 0]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.ExponentialCone()], [1:3])
    r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ 2*exp(1/2)+3 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 2, 2*exp(1/2)] atol=1e-4 rtol=1e-4
    @test r.y ≈ -[1+exp(1/2)/2, 1+exp(1/2)] atol=1e-4 rtol=1e-4
    @test r.z ≈ c+A'*r.y atol=1e-4 rtol=1e-4
end

function _power1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[1, 0, 0, -1, -1, 0]
    A = Float64[1 1 1/2 0 0 0; 0 0 0 0 0 1]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 6, 6)[[4, 1, 2, 5, 3, 6], :]
    h = zeros(6)
    cone = Hypatia.Cone([Hypatia.PowerCone([0.2, 0.8]), Hypatia.PowerCone([0.4, 0.6])], [1:3, 4:6])
    r = solveandcheck(opt, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.pobj ≈ -1.80734 atol=1e-4 rtol=1e-4
    @test r.x[1:3] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
end
