#=
Copyright 2018, Chris Coey and contributors
=#

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
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
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
    @test_throws ErrorException("some primal equality constraints are inconsistent") fullsolve(opt, c, A, b, G, h, cone)
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
    @test_throws ErrorException("some dual equality constraints are inconsistent") fullsolve(opt, c, A, b, G, h, cone)
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
    rnn = fullsolve(opt, c, A, b, G, h, cone)
    @test rnn.status == :Optimal
    @test rnn.pobj ≈ rnn.dobj atol=1e-4 rtol=1e-4

    # nonpositive cone
    opt = Hypatia.Optimizer(verbose=verbose)
    G = SparseMatrixCSC(1.0I, q, n)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q])
    rnp = fullsolve(opt, c, A, b, G, h, cone)
    @test rnp.status == :Optimal
    @test rnp.pobj ≈ rnp.dobj atol=1e-4 rtol=1e-4

    @test rnp.pobj ≈ rnn.pobj atol=1e-4 rtol=1e-4
end

function _orthant2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G*ones(n)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q])
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
end

function _orthant3(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    Random.seed!(1)
    (n, p, q) = (30, 12, 30)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = Diagonal(1.0I, n)
    h = zeros(q)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q])
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
end

function _ellinf1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.EllInfinityCone(3)], [1:3])
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
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
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ 1 atol=1e-4 rtol=1e-4
end

function _soc1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.SecondOrderCone(3)], [1:3])
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 1/sqrt(2), 1/sqrt(2)] atol=1e-4 rtol=1e-4
    @test r.y ≈ [sqrt(2), 0] atol=1e-4 rtol=1e-4
end

function _rsoc1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[0, 0, -1, -1]
    A = Float64[1 0 0 0; 0 1 0 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)
    cone = Hypatia.Cone([Hypatia.RotatedSecondOrderCone(4)], [1:4])
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 15
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
    @test r.x[3:4] ≈ [1, 1]/sqrt(2) atol=1e-4 rtol=1e-4
end

function _rsoc2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[0, 0, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1/2, 1]/sqrt(2)
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.RotatedSecondOrderCone(3)], [1:3])
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -1/sqrt(2) atol=1e-4 rtol=1e-4
    @test r.x[2] ≈ 1/sqrt(2) atol=1e-4 rtol=1e-4
end

function _psd1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[0, -1, 0]
    A = Float64[1 0 0; 0 0 1]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.PositiveSemidefiniteCone(3)], [1:3])
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x[2] ≈ 1 atol=1e-4 rtol=1e-4
end

function _psd2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[1, 0, 1, 0, 0, 1]
    A = Float64[1 2 3 4 5 6; 1 1 1 1 1 1]
    b = Float64[10, 3]
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)
    cone = Hypatia.Cone([Hypatia.PositiveSemidefiniteCone(6)], [1:6])
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ 1.249632 atol=1e-4 rtol=1e-4
    @test r.x ≈ [0.491545, 0.647333, 0.426249, 0.571161, 0.531874, 0.331838] atol=1e-4 rtol=1e-4
end

function _exp1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose)
    c = Float64[1, 1, 1]
    A = Float64[0 1 0; 1 0 0]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.ExponentialCone()], [1:3])
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
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
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -1.80734 atol=1e-4 rtol=1e-4
    @test r.x[1:3] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
end
