#=
Copyright 2018, Chris Coey and contributors
=#
using Hypatia, Test

const verbose = false
const lscachetype = Hypatia.QRSymmCache
SECONDORDER = false

egs_dir = "C:/Users/lkape/.julia/dev/Hypatia/examples"
include(joinpath(egs_dir, "envelope/envelope.jl"))
include(joinpath(egs_dir, "lp/lp.jl"))
include(joinpath(egs_dir, "namedpoly/namedpoly.jl"))

function fullsolve(opt::Hypatia.Optimizer, c, A, b, G, h, cone) # TODO handle lscachetype
    Hypatia.check_data(c, A, b, G, h, cone)
    (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)

    L = Hypatia.QRSymmCache(c1, A1, b1, G1, h, cone, Q2, RiQ1)
    # L = Hypatia.NaiveCache(c1, A1, b1, G1, h, cone)

    Hypatia.load_data!(opt, c1, A1, b1, G1, h, cone, L)

    Hypatia.solve!(opt)

    x = zeros(length(c))
    x[dukeep] = Hypatia.get_x(opt)
    y = zeros(length(b))
    y[prkeep] = Hypatia.get_y(opt)
    s = Hypatia.get_s(opt)
    z = Hypatia.get_z(opt)

    pobj = Hypatia.get_pobj(opt)
    dobj = Hypatia.get_dobj(opt)

    status = Hypatia.get_status(opt)
    stime = Hypatia.get_solvetime(opt)
    niters = Hypatia.get_niters(opt)

    return (x=x, y=y, s=s, z=z, pobj=pobj, dobj=dobj, status=status, stime=stime, niters=niters)
end

function _consistent1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

function _inconsistent1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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

function _orthant1a(verbose::Bool, lscachetype)
    Random.seed!(1)
    (n, p, q) = (40, 20, 40)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    h = zeros(q)

    # nonnegative cone
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    G = SparseMatrixCSC(-1.0I, q, n)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q])
    rnn = fullsolve(opt, c, A, b, G, h, cone)
    @test rnn.status == :Optimal
    @test rnn.pobj ≈ rnn.dobj atol=1e-4 rtol=1e-4
    rnn.niters
end

function _orthant1b(verbose::Bool, lscachetype)
    Random.seed!(1)
    (n, p, q) = (40, 20, 40)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    h = zeros(q)

    # nonpositive cone
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    G = SparseMatrixCSC(1.0I, q, n)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q])
    rnp = fullsolve(opt, c, A, b, G, h, cone)
    @test rnp.status == :Optimal
    @test rnp.pobj ≈ rnp.dobj atol=1e-4 rtol=1e-4

    # @test rnp.pobj ≈ rnn.pobj atol=1e-4 rtol=1e-4
    rnp.niters
    # [nite rs1; niters2]
end

function _orthant2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

function _orthant3(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

function _ellinf1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

function _ellinf2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

function _soc1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

function _rsoc1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    c = Float64[0, 0, -1, -1]
    A = Float64[1 0 0 0; 0 1 0 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)
    cone = Hypatia.Cone([Hypatia.RotatedSecondOrderCone(4)], [1:4])
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
    @test r.x[3:4] ≈ [1, 1]/sqrt(2) atol=1e-4 rtol=1e-4
    r.niters
end

function _rsoc2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

function _psd1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

function _psd2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

function _exp1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

function _power1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
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
    r.niters
end

# ==========================================

function _envelope1a(verbose::Bool, lscachetype)
    # dense methods
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_envelope!(2, 5, 1, 5, use_data=true, dense=true)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -25.502777 atol=1e-4 rtol=1e-4
    @test r.niters <= 35
    r.niters
end

function _envelope1b(verbose::Bool, lscachetype)
    # sparse methods
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_envelope!(2, 5, 1, 5, use_data=true, dense=false)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -25.502777 atol=1e-4 rtol=1e-4
    @test r.niters <= 35
    r.niters
end

function _envelope2a(verbose::Bool, lscachetype)
    # dense methods
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_envelope!(2, 4, 2, 7, dense=true)
    rd = fullsolve(opt, c, A, b, G, h, cone)
    @test rd.status == :Optimal
    @test rd.niters <= 60
    @test rd.pobj ≈ rd.dobj atol=1e-4 rtol=1e-4
    rd.niters
end

function _envelope2b(verbose::Bool, lscachetype)
    # sparse methods
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_envelope!(2, 4, 2, 7, dense=false)
    rs = fullsolve(opt, c, A, b, G, h, cone)
    @test rs.status == :Optimal
    @test rs.niters <= 60
    @test rs.pobj ≈ rs.dobj atol=1e-4 rtol=1e-4

    # @test rs.pobj ≈ rd.pobj atol=1e-4 rtol=1e-4
    rs.niters
end

function _envelope3(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_envelope!(2, 3, 3, 5, dense=false)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    r.niters
end

function _envelope4(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER) # tolrelopt=1e-5, tolabsopt=1e-6, tolfeas=1e-6
    (c, A, b, G, h, cone) = build_envelope!(2, 3, 4, 4, dense=false)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    r.niters
end

function _lp1a(verbose::Bool, lscachetype)
    # dense methods
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_lp!(50, 100, dense=true, tosparse=false)
    rd = fullsolve(opt, c, A, b, G, h, cone)
    @test rd.status == :Optimal
    @test rd.niters <= 45
    @test rd.pobj ≈ rd.dobj atol=1e-4 rtol=1e-4
    rd.niters
end

function _lp1b(verbose::Bool, lscachetype)
    # sparse methods
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_lp!(50, 100, dense=true, tosparse=true)
    rs = fullsolve(opt, c, A, b, G, h, cone)
    @test rs.status == :Optimal
    @test rs.niters <= 45
    @test rs.pobj ≈ rs.dobj atol=1e-4 rtol=1e-4

    # @test rs.pobj ≈ rd.pobj atol=1e-4 rtol=1e-4
    rs.niters
end

function _lp2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_lp!(500, 1000, use_data=true, dense=true)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 90
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ 2055.807 atol=1e-4 rtol=1e-4
    r.niters
end

function _lp3(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_lp!(500, 1000, dense=false, nzfrac=10/1000)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 85
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    r.niters
end

# for namedpoly tests, most optimal values are taken from https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html

function _namedpoly1(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_namedpoly!(:butcher, 2)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 45
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -1.4393333333 atol=1e-4 rtol=1e-4
    r.niters
end

function _namedpoly2(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER) # tolfeas=5e-7
    (c, A, b, G, h, cone) = build_namedpoly!(:caprasse, 4)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 45
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -3.1800966258 atol=1e-4 rtol=1e-4
    r.niters
end

function _namedpoly3(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, tolfeas=1e-10)
    (c, A, b, G, h, cone) = build_namedpoly!(:goldsteinprice, 7)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 60
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ 3 atol=1e-4 rtol=1e-4
    r.niters
end

function _namedpoly4(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_namedpoly!(:heart, 2)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    # @test r.niters <= 40
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -1.36775 atol=1e-4 rtol=1e-4
    r.niters
end

function _namedpoly5(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_namedpoly!(:lotkavolterra, 3)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 35
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -20.8 atol=1e-4 rtol=1e-4
    r.niters
end

function _namedpoly6(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_namedpoly!(:magnetism7, 2)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    # @test r.niters <= 40
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -0.25 atol=1e-4 rtol=1e-4
    r.niters
end

function _namedpoly7(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER) # tolrelopt=1e-5, tolabsopt=1e-6, tolfeas=1e-6
    (c, A, b, G, h, cone) = build_namedpoly!(:motzkin, 7)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 40
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
    r.niters
end

function _namedpoly8(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_namedpoly!(:reactiondiffusion, 4)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 35
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ -36.71269068 atol=1e-4 rtol=1e-4
    r.niters
end

function _namedpoly9(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, secondorder=SECONDORDER)
    (c, A, b, G, h, cone) = build_namedpoly!(:robinson, 8)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 40
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ 0.814814 atol=1e-4 rtol=1e-4
    r.niters
end

function _namedpoly10(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, tolfeas=1e-10)
    (c, A, b, G, h, cone) = build_namedpoly!(:rosenbrock, 3)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 65
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
    r.niters
end

function _namedpoly11(verbose::Bool, lscachetype)
    opt = Hypatia.Optimizer(verbose=verbose, tolfeas=1e-9)
    (c, A, b, G, h, cone) = build_namedpoly!(:schwefel, 4)
    r = fullsolve(opt, c, A, b, G, h, cone)
    @test r.status == :Optimal
    @test r.niters <= 55
    @test r.pobj ≈ r.dobj atol=1e-4 rtol=1e-4
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
    r.niters
end

function get_niters()
    niters = Int[]
    for testfun in (
        _consistent1,
        # _inconsistent1,
        # _inconsistent2,
        _orthant1a,
        _orthant1b,
        _orthant2,
        _orthant3,
        _ellinf1,
        _ellinf2,
        _soc1,
        _rsoc1,
        _rsoc2,
        _psd1,
        _psd2,
        _exp1,
        _power1,
        )
        push!(niters, testfun(verbose, lscachetype))
    end
    for testfun in (
        _envelope1a,
        _envelope1b,
        _envelope2a,
        _envelope2b,
        _envelope3,
        # _envelope4,
        _lp1a,
        _lp1b,
        _lp2,
        _lp3,
        _namedpoly1,
        _namedpoly2,
        _namedpoly3,
        # _namedpoly4,
        _namedpoly5,
        # _namedpoly6,
        _namedpoly7,
        _namedpoly8,
        _namedpoly9,
        # _namedpoly10,
        _namedpoly11,
        )
        push!(niters, testfun(verbose, lscachetype))
    end
    niters
end

itersvec = get_niters()
