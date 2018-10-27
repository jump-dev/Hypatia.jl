#=
Copyright 2018, Chris Coey and contributors
=#

function _dimension1(; verbose, lscachetype)
    A = Matrix{Float64}(undef, 0, 2)
    b = Float64[]
    G = [1.0 0.0]
    h = [1.0]
    c = [-1.0, 0.0]
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(1)], [1:1], [false])

    mdl = Hypatia.Model(verbose=verbose)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
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
    mdl = Hypatia.Model(verbose=verbose)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    r.status == :Optimal
    @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 0] atol=1e-4 rtol=1e-4
    @test isempty(r.y)

    c = [-1.0, -1.0]
    Hypatia.check_data(c, A, b, G, h, cone)
    @test_throws ErrorException("some dual equality constraints are inconsistent") Hypatia.preprocess_data(c, A, b, G, useQR=true)
end

function _consistent1(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
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
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
end

function _inconsistent1(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
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
    @test_throws ErrorException("some primal equality constraints are inconsistent") solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
end

function _inconsistent2(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
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
    @test_throws ErrorException("some dual equality constraints are inconsistent") solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
end

function _orthant1(; verbose, lscachetype)
    Random.seed!(1)
    (n, p, q) = (6, 3, 6)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    h = zeros(q)

    # nonnegative cone
    mdl = Hypatia.Model(verbose=verbose)
    G = SparseMatrixCSC(-1.0I, q, n)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q])
    rnn = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test rnn.status == :Optimal

    # nonpositive cone
    mdl = Hypatia.Model(verbose=verbose)
    G = SparseMatrixCSC(1.0I, q, n)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q])
    rnp = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test rnp.status == :Optimal

    @test rnp.pobj ≈ rnn.pobj atol=1e-4 rtol=1e-4
end

function _orthant2(; verbose, lscachetype)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G*ones(n)

    mdl = Hypatia.Model(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q], [true])
    r1 = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r1.status == :Optimal

    mdl = Hypatia.Model(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(q)], [1:q], [false])
    r2 = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r2.status == :Optimal

    @test r1.pobj ≈ r2.pobj atol=1e-4 rtol=1e-4
end

function _orthant3(; verbose, lscachetype)
    Random.seed!(1)
    (n, p, q) = (15, 6, 15)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = Diagonal(1.0I, n)
    h = zeros(q)

    mdl = Hypatia.Model(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q], [true])
    r1 = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r1.status == :Optimal

    mdl = Hypatia.Model(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonpositiveCone(q)], [1:q], [false])
    r2 = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r2.status == :Optimal

    @test r1.pobj ≈ r2.pobj atol=1e-4 rtol=1e-4
end

function _orthant4(; verbose, lscachetype)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A*ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G*ones(n)

    mdl = Hypatia.Model(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(4), Hypatia.NonnegativeCone(6)], [1:4, 5:10], [false, true])
    r1 = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r1.status == :Optimal

    mdl = Hypatia.Model(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.NonnegativeCone(10)], [1:10], [false])
    r2 = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r2.status == :Optimal

    @test r1.pobj ≈ r2.pobj atol=1e-4 rtol=1e-4
end

function _epinorminf1(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.EpiNormInf(3)], [1:3])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ -1 - 1/sqrt(2) atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 1/sqrt(2), 1] atol=1e-4 rtol=1e-4
    @test r.y ≈ [1, 1] atol=1e-4 rtol=1e-4
end

function _epinorminf2(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A*ones(6)
    G = rand(6, 6)
    h = G*ones(6)
    cone = Hypatia.Cone([Hypatia.EpiNormInf(6)], [1:6])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ 1 atol=1e-4 rtol=1e-4
end

function _epinorminf3(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = zeros(0, 6)
    b = zeros(0)
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)
    cone = Hypatia.Cone([Hypatia.EpiNormInf(6)], [1:6])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
    @test r.x ≈ zeros(6) atol=1e-4 rtol=1e-4
end

function _epinorminf4(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    c = Float64[0, 1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, -0.4]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.EpiNormInf(3)], [1:3], [true])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, -0.4, 0.6] atol=1e-4 rtol=1e-4
    @test r.y ≈ [1, 0] atol=1e-4 rtol=1e-4
end

function _epinorminf5(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    Random.seed!(1)
    c = Float64[1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A*ones(6)
    G = rand(6, 6)
    h = G*ones(6)
    cone = Hypatia.Cone([Hypatia.EpiNormInf(6)], [1:6], [true])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 15
    @test r.pobj ≈ 1 atol=1e-4 rtol=1e-4
end

function _epinorminf6(; verbose, lscachetype)
    Random.seed!(1)
    l = 3
    L = 2l + 1
    c = collect(-Float64(l):Float64(l))
    A = spzeros(2, L)
    A[1,1] = A[1,L] = A[2,1] = 1.0; A[2,L] = -1.0
    b = [0.0, 0.0]
    G = [spzeros(1, L); sparse(1.0I, L, L); spzeros(1, L); sparse(2.0I, L, L)]
    h = zeros(2L+2); h[1] = 1.0; h[L+2] = 1.0
    mdl = Hypatia.Model(verbose=verbose)
    cone = Hypatia.Cone([Hypatia.EpiNormInf(L+1), Hypatia.EpiNormInf(L+1)], [1:L+1, L+2:2L+2], [true, false])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.pobj ≈ -l + 1 atol=1e-4 rtol=1e-4
    @test r.x[2] ≈ 0.5 atol=1e-4 rtol=1e-4
    @test r.x[end-1] ≈ -0.5 atol=1e-4 rtol=1e-4
    @test sum(abs, r.x) ≈ 1.0 atol=1e-4 rtol=1e-4
end

function _epinormeucl1(; verbose, lscachetype)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1, 1/sqrt(2)]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.EpiNormEucl(3)], [1:3], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x ≈ [1, 1/sqrt(2), 1/sqrt(2)] atol=1e-4 rtol=1e-4
        @test r.y ≈ [sqrt(2), 0] atol=1e-4 rtol=1e-4
    end
end

function _epinormeucl2(; verbose, lscachetype)
    c = Float64[0, -1, -1]
    A = Float64[1 0 0]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.EpiNormEucl(3)], [1:3], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(3) atol=1e-4 rtol=1e-4
    end
end

function _epipersquare1(; verbose, lscachetype)
    c = Float64[0, 0, -1, -1]
    A = Float64[1 0 0 0; 0 1 0 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.EpiPerSquare(4)], [1:4], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x[3:4] ≈ [1, 1]/sqrt(2) atol=1e-4 rtol=1e-4
    end
end

function _epipersquare2(; verbose, lscachetype)
    c = Float64[0, 0, -1]
    A = Float64[1 0 0; 0 1 0]
    b = Float64[1/2, 1]/sqrt(2)
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.EpiPerSquare(3)], [1:3], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 15
        @test r.pobj ≈ -1/sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x[2] ≈ 1/sqrt(2) atol=1e-4 rtol=1e-4
    end
end

function _epipersquare3(; verbose, lscachetype)
    c = Float64[0, 1, -1, -1]
    A = Float64[1 0 0 0]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.EpiPerSquare(4)], [1:4], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(4) atol=1e-4 rtol=1e-4
    end
end

function _semidefinite1(; verbose, lscachetype)
    c = Float64[0, -1, 0]
    A = Float64[1 0 0; 0 0 1]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.PosSemidef(3)], [1:3], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ -1 atol=1e-4 rtol=1e-4
        @test r.x[2] ≈ 1 atol=1e-4 rtol=1e-4
    end
end

function _semidefinite2(; verbose, lscachetype)
    c = Float64[0, -1, 0]
    A = Float64[1 0 1]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.PosSemidef(3)], [1:3], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(3) atol=1e-4 rtol=1e-4
    end
end

function _semidefinite3(; verbose, lscachetype)
    c = Float64[1, 0, 1, 0, 0, 1]
    A = Float64[1 2 3 4 5 6; 1 1 1 1 1 1]
    b = Float64[10, 3]
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.PosSemidef(6)], [1:6], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ 1.249632 atol=1e-4 rtol=1e-4
        @test r.x ≈ [0.491545, 0.647333, 0.426249, 0.571161, 0.531874, 0.331838] atol=1e-4 rtol=1e-4
    end
end

function _hypoperlog1(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    c = Float64[1, 1, 1]
    A = Float64[0 1 0; 1 0 0]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.HypoPerLog()], [1:3])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ 2*exp(1/2)+3 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 2, 2*exp(1/2)] atol=1e-4 rtol=1e-4
    @test r.y ≈ -[1+exp(1/2)/2, 1+exp(1/2)] atol=1e-4 rtol=1e-4
    @test r.z ≈ c+A'*r.y atol=1e-4 rtol=1e-4
end

function _hypoperlog2(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    c = Float64[-1, 0, 0]
    A = Float64[0 1 0]
    b = Float64[0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.HypoPerLog()], [1:3])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 20
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
end

function _hypoperlog3(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    c = Float64[1, 1, 1]
    A = Matrix{Float64}(undef, 0, 3)
    b = Vector{Float64}(undef, 0)
    G = sparse([1, 2, 3, 4], [1, 2, 3, 1], -ones(4))
    h = zeros(4)
    cone = Hypatia.Cone([Hypatia.HypoPerLog(), Hypatia.NonnegativeCone(1)], [1:3, 4:4])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 15
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
    @test r.x ≈ [0, 0, 0] atol=1e-4 rtol=1e-4
    @test isempty(r.y)
end

function _hypoperlog4(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    c = Float64[0, 0, 1]
    A = Float64[0 1 0; 1 0 0]
    b = Float64[1, -1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cone = Hypatia.Cone([Hypatia.HypoPerLog()], [1:3], [true])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 15
    @test r.pobj ≈ exp(-2) atol=1e-4 rtol=1e-4
    @test r.x ≈ [-1, 1, exp(-2)] atol=1e-4 rtol=1e-4
end

function _hypogeomean1(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    c = Float64[1, 0, 0, -1, -1, 0]
    A = Float64[1 1 1/2 0 0 0; 0 0 0 0 0 1]
    b = Float64[2, 1]
    G = SparseMatrixCSC(-1.0I, 6, 6)[[4, 1, 2, 5, 3, 6], :]
    h = zeros(6)
    cone = Hypatia.Cone([Hypatia.HypoGeomean([0.2, 0.8]), Hypatia.HypoGeomean([0.4, 0.6])], [1:3, 4:6])
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 25
    @test r.pobj ≈ -1.80734 atol=1e-4 rtol=1e-4
    @test r.x[1:3] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
end

function _hypogeomean2(; verbose, lscachetype)
    c = Float64[-1, 0, 0]
    A = Float64[0 0 1; 0 1 0]
    b = Float64[1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.HypoGeomean([0.5, 0.5])], [1:3], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ (usedual ? 0 : -1/sqrt(2)) atol=1e-4 rtol=1e-4
        @test r.x[2:3] ≈ [1, 0.5] atol=1e-4 rtol=1e-4
    end
end

function _hypogeomean3(; verbose, lscachetype)
    l = 4
    c = vcat(0.0, ones(l))
    A = [1.0 zeros(1, l)]
    G = SparseMatrixCSC(-1.0I, l+1, l+1)
    h = zeros(l+1)

    for usedual in [true, false]
        b = (usedual ? [-1.0] : [1.0])
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.HypoGeomean(fill(1/l, l))], [1:l+1], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        @test r.pobj ≈ (usedual ? 1 : l) atol=1e-4 rtol=1e-4
        @test r.x[2:end] ≈ (usedual ? fill(1/l, l) : ones(l)) atol=1e-4 rtol=1e-4
    end
end

function _hypogeomean4(; verbose, lscachetype)
    l = 4
    c = ones(l)
    A = zeros(0, l)
    b = zeros(0)
    G = [zeros(1, l); Matrix(-1.0I, l, l)]
    h = zeros(l+1)

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.HypoGeomean(fill(1/l, l))], [1:l+1], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 15
        @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(l) atol=1e-4 rtol=1e-4
    end
end

function _spectral1(; verbose, lscachetype)
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    Xnm = Xn*Xm
    c = vcat(1.0, zeros(Xnm))
    p = 0
    A = [spzeros(Xnm, 1) sparse(1.0I, Xnm, Xnm)]
    b = rand(Xnm)
    G = sparse(-1.0I, Xnm+1, Xnm+1)
    h = vcat(0.0, rand(Xnm))

    for usedual in [true, false]
        mdl = Hypatia.Model(verbose=verbose)
        cone = Hypatia.Cone([Hypatia.EpiNormSpectral(Xnm+1, Xn, Xm)], [1:Xnm+1], [usedual])
        r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
        @test r.status == :Optimal
        @test r.niters <= 20
        if usedual
            @test sum(svdvals!(reshape(r.s[2:end], Xn, Xm))) ≈ r.pobj atol=1e-4 rtol=1e-4
            @test svdvals!(reshape(r.z[2:end], Xn, Xm))[1] ≈ r.z[1] atol=1e-4 rtol=1e-4
        else
            @test svdvals!(reshape(r.s[2:end], Xn, Xm))[1] ≈ r.pobj atol=1e-4 rtol=1e-4
            @test sum(svdvals!(reshape(r.z[2:end], Xn, Xm))) ≈ r.z[1] atol=1e-4 rtol=1e-4
        end
    end
end
