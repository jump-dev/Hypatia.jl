#=
Copyright 2018, Chris Coey and contributors
=#

function solveandcheck(model::MO.Model, solver::SO.Solver; atol=1e-4, rtol=1e-4)
    SO.solve(solver)

    x = SO.get_x(solver)
    y = SO.get_y(solver)
    s = SO.get_s(solver)
    z = SO.get_z(solver)
    primal_obj = SO.get_primal_obj(solver)
    dual_obj = SO.get_dual_obj(solver)
    status = SO.get_status(solver)
    solve_time = SO.get_solve_time(solver)

    # check conic certificates are valid
    (c, A, b, G, h) = (model.c, model.A, model.b, model.G, model.h)
    if status == :Optimal
        @test primal_obj ≈ dual_obj atol=atol rtol=rtol
        @test A * x ≈ b atol=atol rtol=rtol
        @test G * x + s ≈ h atol=atol rtol=rtol
        @test G' * z + A' * y ≈ -c atol=atol rtol=rtol
        @test dot(s, z) ≈ 0.0 atol=atol rtol=rtol
        @test dot(c, x) ≈ primal_obj atol=1e-8 rtol=1e-8
        @test dot(b, y) + dot(h, z) ≈ -dual_obj atol=1e-8 rtol=1e-8
    elseif status == :PrimalInfeasible
        @test isnan(primal_obj)
        @test dual_obj > 0
        @test dot(b, y) + dot(h, z) ≈ -dual_obj atol=1e-8 rtol=1e-8
        @test G' * z ≈ -A' * y atol=atol rtol=rtol
    elseif status == :DualInfeasible
        @test isnan(dual_obj)
        @test primal_obj < 0
        @test dot(c, x) ≈ primal_obj atol=1e-8 rtol=1e-8
        @test G * x ≈ -s atol=atol rtol=rtol
        @test A * x ≈ zeros(length(y)) atol=atol rtol=rtol
    elseif status == :IllPosed
        # TODO primal vs dual ill-posed statuses and conditions
    end

    return (x=x, y=y, s=s, z=z, primal_obj=primal_obj, dual_obj=dual_obj, status=status, solve_time=solve_time)
end

function dimension1(; verbose::Bool = true)
    c = [-1, 0]
    A = Matrix{Float64}(undef, 0, 2)
    b = []
    G = [1 0]
    h = [1]
    cones = [CO.Nonnegative(1, false)]
    cone_idxs = [1:1]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    for dense in (true, false)
        if !dense
            model.A = sparse(A)
            model.G = sparse(G)
        end

        r = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
        r.status == :Optimal
        @test r.primal_obj ≈ -1 atol=1e-4 rtol=1e-4
        @test r.x ≈ [1, 0] atol=1e-4 rtol=1e-4
        @test isempty(r.y)

        # TODO
        # model.c = [-1.0, -1.0]
        # @test_throws ErrorException("some dual equality constraints are inconsistent") HYP.preprocess_data(c, A, b, G, useQR = true)
    end
end

# function consistent1(; verbose::Bool = true)
#     model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
#     Random.seed!(1)
#     (n, p, q) = (30, 15, 30)
#     A = rand(-9.0:9.0, p, n)
#     G = Matrix(1.0I, q, n)
#     c = rand(0.0:9.0, n)
#     rnd1 = rand()
#     rnd2 = rand()
#     A[11:15, :] = rnd1*A[1:5, :] - rnd2*A[6:10, :]
#     b = A * ones(n)
#     rnd1 = rand()
#     rnd2 = rand()
#     A[:,11:15] = rnd1*A[:,1:5] - rnd2*A[:,6:10]
#     G[:,11:15] = rnd1*G[:,1:5] - rnd2*G[:,6:10]
#     c[11:15] = rnd1*c[1:5] - rnd2*c[6:10]
#     h = zeros(q)
#     cone = CO.Cone([CO.Nonpositive(q)], [1:q])
#     solver = SO.HSDSolver(model, verbose = verbose)
    # r = solveandcheck(model, solver)
#     @test r.status == :Optimal
# end
#
# function inconsistent1(; verbose::Bool = true)
#     model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
#     Random.seed!(1)
#     (n, p, q) = (30, 15, 30)
#     A = rand(-9.0:9.0, p, n)
#     G = Matrix(-1.0I, q, n)
#     c = rand(0.0:9.0, n)
#     b = rand(p)
#     rnd1 = rand()
#     rnd2 = rand()
#     A[11:15, :] = rnd1*A[1:5, :] - rnd2*A[6:10, :]
#     b[11:15] = 2*(rnd1*b[1:5] - rnd2*b[6:10])
#     h = zeros(q)
#     cone = CO.Cone([CO.Nonnegative(q)], [1:q])
#     @test_throws ErrorException("some primal equality constraints are inconsistent") solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
# end
#
# function inconsistent2(; verbose::Bool = true)
#     model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
#     Random.seed!(1)
#     (n, p, q) = (30, 15, 30)
#     A = rand(-9.0:9.0, p, n)
#     G = Matrix(-1.0I, q, n)
#     c = rand(0.0:9.0, n)
#     b = rand(p)
#     rnd1 = rand()
#     rnd2 = rand()
#     A[:,11:15] = rnd1*A[:,1:5] - rnd2*A[:,6:10]
#     G[:,11:15] = rnd1*G[:,1:5] - rnd2*G[:,6:10]
#     c[11:15] = 2*(rnd1*c[1:5] - rnd2*c[6:10])
#     h = zeros(q)
#     cone = CO.Cone([CO.Nonnegative(q)], [1:q])
#     @test_throws ErrorException("some dual equality constraints are inconsistent") solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
# end

function orthant1(; verbose::Bool = true)
    Random.seed!(1)
    (n, p, q) = (6, 3, 6)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A * ones(n)
    h = zeros(q)

    # nonnegative cone
    model = MO.Linear(c, A, b, SparseMatrixCSC(-1.0I, q, n), h, [CO.Nonnegative(q)], [1:q])
    rnn = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test rnn.status == :Optimal

    # nonpositive cone
    model = MO.Linear(c, A, b, SparseMatrixCSC(1.0I, q, n), h, [CO.Nonpositive(q)], [1:q])
    rnp = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test rnp.status == :Optimal

    @test rnp.primal_obj ≈ rnn.primal_obj atol=1e-4 rtol=1e-4
end

function orthant2(; verbose::Bool = true)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A * ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G * ones(n)

    # use dual barrier
    model = MO.Linear(c, A, b, G, h, [CO.Nonnegative(q, true)], [1:q])
    r1 = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test r1.status == :Optimal

    # use primal barrier
    model = MO.Linear(c, A, b, G, h, [CO.Nonnegative(q, false)], [1:q])
    r2 = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=1e-4 rtol=1e-4
end

function orthant3(; verbose::Bool = true)
    Random.seed!(1)
    (n, p, q) = (15, 6, 15)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A * ones(n)
    G = Diagonal(1.0I, n)
    h = zeros(q)

    # use dual barrier
    model = MO.Linear(c, A, b, G, h, [CO.Nonpositive(q, true)], [1:q])
    r1 = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test r1.status == :Optimal

    # use primal barrier
    model = MO.Linear(c, A, b, G, h, [CO.Nonpositive(q, false)], [1:q])
    r2 = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=1e-4 rtol=1e-4
end

function orthant4(; verbose::Bool = true)
    Random.seed!(1)
    (n, p, q) = (5, 2, 10)
    c = rand(0.0:9.0, n)
    A = rand(-9.0:9.0, p, n)
    b = A * ones(n)
    G = rand(q, n) - Matrix(2.0I, q, n)
    h = G * ones(n)

    # mixture of nonnegative and nonpositive cones
    model = MO.Linear(c, A, b, G, h, [CO.Nonnegative(4, false), CO.Nonnegative(6, true)], [1:4, 5:10])
    r1 = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test r1.status == :Optimal

    # only nonnegative cone
    model = MO.Linear(c, A, b, G, h, [CO.Nonnegative(10, false)], [1:10])
    r2 = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test r2.status == :Optimal

    @test r1.primal_obj ≈ r2.primal_obj atol=1e-4 rtol=1e-4
end

function epinorminf1(; verbose::Bool = true)
    c = [0, -1, -1]
    A = [1 0 0; 0 1 0]
    b = [1, inv(sqrt(2))]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cones = [CO.EpiNormInf(3)]
    cone_idxs = [1:3]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 - inv(sqrt(2)) atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, inv(sqrt(2)), 1] atol=1e-4 rtol=1e-4
    @test r.y ≈ [1, 1] atol=1e-4 rtol=1e-4
end

function epinorminf2(; verbose::Bool = true)
    Random.seed!(1)
    c = [1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A * ones(6)
    G = rand(6, 6)
    h = G * ones(6)
    cones = [CO.EpiNormInf(6)]
    cone_idxs = [1:6]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 1 atol=1e-4 rtol=1e-4
end

function epinorminf3(; verbose::Bool = true)
    Random.seed!(1)
    c = [1, 0, 0, 0, 0, 0]
    A = zeros(0, 6)
    b = zeros(0)
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)
    cones = [CO.EpiNormInf(6)]
    cone_idxs = [1:6]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
    @test r.x ≈ zeros(6) atol=1e-4 rtol=1e-4
end

function epinorminf4(; verbose::Bool = true)
    c = [0, 1, -1]
    A = [1 0 0; 0 1 0]
    b = [1, -0.4]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cones = [CO.EpiNormInf(3, true)]
    cone_idxs = [1:3]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, -0.4, 0.6] atol=1e-4 rtol=1e-4
    @test r.y ≈ [1, 0] atol=1e-4 rtol=1e-4
end

function epinorminf5(; verbose::Bool = true)
    Random.seed!(1)
    c = [1, 0, 0, 0, 0, 0]
    A = rand(-9.0:9.0, 3, 6)
    b = A * ones(6)
    G = rand(6, 6)
    h = G * ones(6)
    cones = [CO.EpiNormInf(6, true)]
    cone_idxs = [1:6]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 1 atol=1e-4 rtol=1e-4
end

function epinorminf6(; verbose::Bool = true)
    Random.seed!(1)
    l = 3
    L = 2l + 1
    c = collect(-l:l)
    A = spzeros(2, L)
    A[1, 1] = A[1, L] = A[2, 1] = 1.0; A[2, L] = -1.0
    b = [0.0, 0.0]
    G = [spzeros(1, L); sparse(1.0I, L, L); spzeros(1, L); sparse(2.0I, L, L)]
    h = zeros(2L + 2); h[1] = 1.0; h[L + 2] = 1.0
    cones = [CO.EpiNormInf(L + 1, true), CO.EpiNormInf(L + 1, false)]
    cone_idxs = [1:(L + 1), (L + 2):(2L + 2)]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -l + 1 atol=1e-4 rtol=1e-4
    @test r.x[2] ≈ 0.5 atol=1e-4 rtol=1e-4
    @test r.x[end - 1] ≈ -0.5 atol=1e-4 rtol=1e-4
    @test sum(abs, r.x) ≈ 1.0 atol=1e-4 rtol=1e-4
end

function epinormeucl1(; verbose::Bool = true)
    c = [0, -1, -1]
    A = [1 0 0; 0 1 0]
    b = [1, inv(sqrt(2))]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.EpiNormEucl(3, is_dual)], [1:3])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x ≈ [1, inv(sqrt(2)), inv(sqrt(2))] atol=1e-4 rtol=1e-4
        @test r.y ≈ [sqrt(2), 0] atol=1e-4 rtol=1e-4
    end
end

function epinormeucl2(; verbose::Bool = true)
    c = [0, -1, -1]
    A = [1 0 0]
    b = [0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.EpiNormEucl(3, is_dual)], [1:3])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(3) atol=1e-4 rtol=1e-4
    end
end

function epipersquare1(; verbose::Bool = true)
    c = [0, 0, -1, -1]
    A = [1 0 0 0; 0 1 0 0]
    b = [1/2, 1]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.EpiPerSquare(4, is_dual)], [1:4])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -sqrt(2) atol=1e-4 rtol=1e-4
        @test r.x[3:4] ≈ [1, 1] / sqrt(2) atol=1e-4 rtol=1e-4
    end
end

function epipersquare2(; verbose::Bool = true)
    c = [0, 0, -1]
    A = [1 0 0; 0 1 0]
    b = [1/2, 1] / sqrt(2)
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.EpiPerSquare(3, is_dual)], [1:3])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -inv(sqrt(2)) atol=1e-4 rtol=1e-4
        @test r.x[2] ≈ inv(sqrt(2)) atol=1e-4 rtol=1e-4
    end
end

function epipersquare3(; verbose::Bool = true)
    c = [0, 1, -1, -1]
    A = [1 0 0 0]
    b = [0]
    G = SparseMatrixCSC(-1.0I, 4, 4)
    h = zeros(4)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.EpiPerSquare(4, is_dual)], [1:4])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(4) atol=1e-4 rtol=1e-4
    end
end

function semidefinite1(; verbose::Bool = true)
    c = [0, -1, 0]
    A = [1 0 0; 0 0 1]
    b = [1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.PosSemidef(3, is_dual)], [1:3])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ -1 atol=1e-4 rtol=1e-4
        @test r.x[2] ≈ 1 atol=1e-4 rtol=1e-4
    end
end

function semidefinite2(; verbose::Bool = true)
    c = [0, -1, 0]
    A = [1 0 1]
    b = [0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.PosSemidef(3, is_dual)], [1:3])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(3) atol=1e-4 rtol=1e-4
    end
end

function semidefinite3(; verbose::Bool = true)
    c = [1, 0, 1, 0, 0, 1]
    A = [1 2 3 4 5 6; 1 1 1 1 1 1]
    b = [10, 3]
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.PosSemidef(6, is_dual)], [1:6])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 1.249632 atol=1e-4 rtol=1e-4
        @test r.x ≈ [0.491545, 0.647333, 0.426249, 0.571161, 0.531874, 0.331838] atol=1e-4 rtol=1e-4
    end
end

function hypoperlog1(; verbose::Bool = true)
    c = [1, 1, 1]
    A = [0 1 0; 1 0 0]
    b = [2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cones = [CO.HypoPerLog()]
    cone_idxs = [1:3]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    ehalf = exp(1 / 2)
    @test r.primal_obj ≈ 2 * ehalf + 3 atol=1e-4 rtol=1e-4
    @test r.x ≈ [1, 2, 2 * ehalf] atol=1e-4 rtol=1e-4
    @test r.y ≈ -[1 + ehalf / 2, 1 + ehalf] atol=1e-4 rtol=1e-4
    @test r.z ≈ c + A' * r.y atol=1e-4 rtol=1e-4
end

function hypoperlog2(; verbose::Bool = true)
    c = [-1, 0, 0]
    A = [0 1 0]
    b = [0]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cones = [CO.HypoPerLog()]
    cone_idxs = [1:3]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
end

function hypoperlog3(; verbose::Bool = true)
    c = [1, 1, 1]
    A = Matrix{Float64}(undef, 0, 3)
    b = Vector{Float64}(undef, 0)
    G = sparse([1, 2, 3, 4], [1, 2, 3, 1], -ones(4))
    h = zeros(4)
    cones = [CO.HypoPerLog(), CO.Nonnegative(1)]
    cone_idxs = [1:3, 4:4]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
    @test r.x ≈ [0, 0, 0] atol=1e-4 rtol=1e-4
    @test isempty(r.y)
end

function hypoperlog4(; verbose::Bool = true)
    c = [0, 0, 1]
    A = [0 1 0; 1 0 0]
    b = [1, -1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)
    cones = [CO.HypoPerLog(true)]
    cone_idxs = [1:3]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ exp(-2) atol=1e-4 rtol=1e-4
    @test r.x ≈ [-1, 1, exp(-2)] atol=1e-4 rtol=1e-4
end

function epiperpower1(; verbose::Bool = true)
    c = [1, 0, -1, 0, 0, -1]
    A = [1 1 0 1/2 0 0; 0 0 0 0 1 0]
    b = [2, 1]
    G = SparseMatrixCSC(-1.0I, 6, 6)
    h = zeros(6)
    cones = [CO.EpiPerPower(5.0, false), CO.EpiPerPower(2.5, false)]
    cone_idxs = [1:3, 4:6]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1.80734 atol=1e-4 rtol=1e-4
    @test r.x[[1, 2, 4]] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
end

function epiperpower2(; verbose::Bool = true)
    c = [0, 0, -1]
    A = [1 0 0; 0 1 0]
    b = [1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.EpiPerPower(2.0, is_dual)], [1:3])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? -sqrt(2) : -inv(sqrt(2))) atol=1e-4 rtol=1e-4
        @test r.x[1:2] ≈ [1/2, 1] atol=1e-4 rtol=1e-4
    end
end

function epiperpower3(; verbose::Bool = true)
    c = [0, 0, 1]
    A = [1 0 0; 0 1 0]
    b = [0, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.EpiPerPower(2.0, is_dual)], [1:3])

        solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-9)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x[1:2] ≈ [0, 1] atol=1e-4 rtol=1e-4
    end
end

function hypogeomean1(; verbose::Bool = true)
    c = [1, 0, 0, -1, -1, 0]
    A = [1 1 1/2 0 0 0; 0 0 0 0 0 1]
    b = [2, 1]
    G = SparseMatrixCSC(-1.0I, 6, 6)[[4, 1, 2, 5, 3, 6], :]
    h = zeros(6)
    cones = [CO.HypoGeomean([0.2, 0.8], false), CO.HypoGeomean([0.4, 0.6], false)]
    cone_idxs = [1:3, 4:6]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ -1.80734 atol=1e-4 rtol=1e-4
    @test r.x[1:3] ≈ [0.0639314, 0.783361, 2.30542] atol=1e-4 rtol=1e-4
end

function hypogeomean2(; verbose::Bool = true)
    c = [-1, 0, 0]
    A = [0 0 1; 0 1 0]
    b = [1/2, 1]
    G = SparseMatrixCSC(-1.0I, 3, 3)
    h = zeros(3)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.HypoGeomean([0.5, 0.5], is_dual)], [1:3])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 0 : -inv(sqrt(2))) atol=1e-4 rtol=1e-4
        @test r.x[2:3] ≈ [1, 0.5] atol=1e-4 rtol=1e-4
    end
end

function hypogeomean3(; verbose::Bool = true)
    l = 4
    c = vcat(0.0, ones(l))
    A = [1.0 zeros(1, l)]
    G = SparseMatrixCSC(-1.0I, l + 1, l + 1)
    h = zeros(l + 1)

    for is_dual in (true, false)
        b = (is_dual ? [-1] : [1])
        model = MO.Linear(c, A, b, G, h, [CO.HypoGeomean(fill(inv(l), l), is_dual)], [1:(l + 1)])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ (is_dual ? 1.0 : l) atol=1e-4 rtol=1e-4
        @test r.x[2:end] ≈ (is_dual ? inv(l) : 1.0) * ones(l) atol=1e-4 rtol=1e-4
    end
end

function hypogeomean4(; verbose::Bool = true)
    l = 4
    c = ones(l)
    A = zeros(0, l)
    b = zeros(0)
    G = [zeros(1, l); Matrix(-1.0I, l, l)]
    h = zeros(l + 1)

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.HypoGeomean(fill(inv(l), l), is_dual)], [1:(l + 1)])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        @test r.primal_obj ≈ 0 atol=1e-4 rtol=1e-4
        @test r.x ≈ zeros(l) atol=1e-4 rtol=1e-4
    end
end

function epinormspectral1(; verbose::Bool = true)
    Random.seed!(1)
    (Xn, Xm) = (3, 4)
    Xnm = Xn * Xm
    c = vcat(1.0, zeros(Xnm))
    A = [spzeros(Xnm, 1) sparse(1.0I, Xnm, Xnm)]
    b = rand(Xnm)
    G = sparse(-1.0I, Xnm + 1, Xnm + 1)
    h = vcat(0.0, rand(Xnm))

    for is_dual in (true, false)
        model = MO.Linear(c, A, b, G, h, [CO.EpiNormSpectral(Xn, Xm, is_dual)], [1:(Xnm + 1)])
        solver = SO.HSDSolver(model, verbose = verbose)
        r = solveandcheck(model, solver)
        @test r.status == :Optimal
        if is_dual
            @test sum(svdvals!(reshape(r.s[2:end], Xn, Xm))) ≈ r.s[1] atol=1e-4 rtol=1e-4
            @test svdvals!(reshape(r.z[2:end], Xn, Xm))[1] ≈ r.z[1] atol=1e-4 rtol=1e-4
        else
            @test svdvals!(reshape(r.s[2:end], Xn, Xm))[1] ≈ r.s[1] atol=1e-4 rtol=1e-4
            @test sum(svdvals!(reshape(r.z[2:end], Xn, Xm))) ≈ r.z[1] atol=1e-4 rtol=1e-4
        end
    end
end

function hypoperlogdet1(; verbose::Bool = true)
    Random.seed!(1)
    side = 4
    dim = 2 + div(side * (side + 1), 2)
    c = [-1.0, 0.0]
    A = [0.0 1.0]
    b = [1.0]
    G = SparseMatrixCSC(-1.0I, dim, 2)
    mat_half = randn(side, side)
    mat = mat_half * mat_half'
    h = zeros(dim)
    CO.smat_to_svec!(view(h, 3:dim), mat)
    cones = [CO.HypoPerLogdet(dim)]
    cone_idxs = [1:dim]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.x[1] ≈ -r.primal_obj atol=1e-4 rtol=1e-4
    @test r.x[2] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2] * logdet(CO.svec_to_smat!(zeros(side, side), r.s[3:end]) / r.s[2]) ≈ r.s[1] atol=1e-4 rtol=1e-4
    @test r.z[1] * (logdet(CO.svec_to_smat!(zeros(side, side), -r.z[3:end]) / r.z[1]) + side) ≈ r.z[2] atol=1e-4 rtol=1e-4
end

function hypoperlogdet2(; verbose::Bool = true)
    Random.seed!(1)
    side = 3
    dim = 2 + div(side * (side + 1), 2)
    c = [0.0, 1.0]
    A = [1.0 0.0]
    b = [-1.0]
    G = SparseMatrixCSC(-1.0I, dim, 2)
    mat_half = rand(side, side)
    mat = mat_half * mat_half'
    h = zeros(dim)
    CO.smat_to_svec!(view(h, 3:dim), mat)
    cones = [CO.HypoPerLogdet(dim, true)]
    cone_idxs = [1:dim]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.x[2] ≈ r.primal_obj atol=1e-4 rtol=1e-4
    @test r.x[1] ≈ -1 atol=1e-4 rtol=1e-4
    @test r.s[1] * (logdet(CO.svec_to_smat!(zeros(side, side), -r.s[3:end]) / r.s[1]) + side) ≈ r.s[2] atol=1e-4 rtol=1e-4
    @test r.z[2] * logdet(CO.svec_to_smat!(zeros(side, side), r.z[3:end]) / r.z[2]) ≈ r.z[1] atol=1e-4 rtol=1e-4
end

function hypoperlogdet3(; verbose::Bool = true)
    Random.seed!(1)
    side = 3
    dim = 2 + div(side * (side + 1), 2)
    c = [-1.0, 0.0]
    A = [0.0 1.0]
    b = [0.0]
    G = SparseMatrixCSC(-1.0I, dim, 2)
    mat_half = rand(side, side)
    mat = mat_half * mat_half'
    h = zeros(dim)
    CO.smat_to_svec!(view(h, 3:dim), mat)
    cones = [CO.HypoPerLogdet(dim)]
    cone_idxs = [1:dim]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.x[1] ≈ -r.primal_obj atol=1e-4 rtol=1e-4
    @test r.x ≈ [0, 0] atol=1e-4 rtol=1e-4
end

function epipersumexp1(; verbose::Bool = true)
    l = 5
    c = vcat(0.0, -ones(l))
    A = [1.0 zeros(1, l)]
    b = [1.0]
    G = [-1.0 spzeros(1, l); spzeros(1, l + 1); spzeros(l, 1) sparse(-1.0I, l, l)]
    h = zeros(l + 2)
    cones = [CO.EpiPerSumExp(l + 2)]
    cone_idxs = [1:(l + 2)]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.x[1] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2] ≈ 0 atol=1e-4 rtol=1e-4
    @test r.s[1] ≈ 1 atol=1e-4 rtol=1e-4
end

function epipersumexp2(; verbose::Bool = true)
    l = 5
    c = vcat(0.0, -ones(l))
    A = [1.0 zeros(1, l)]
    b = [1.0]
    G = [-1.0 spzeros(1, l); spzeros(1, l + 1); spzeros(l, 1) sparse(-1.0I, l, l)]
    h = zeros(l + 2); h[2] = 1.0
    cones = [CO.EpiPerSumExp(l + 2)]
    cone_idxs = [1:(l + 2)]
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)

    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.x[1] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2] ≈ 1 atol=1e-4 rtol=1e-4
    @test r.s[2] * sum(exp, r.s[3:end] / r.s[2]) ≈ r.s[1] atol=1e-4 rtol=1e-4
end

function envelope1(; verbose::Bool = true)
    # dense methods
    (c, A, b, G, h, cones, cone_idxs) = build_envelope(2, 5, 1, 5, use_data = true, dense = true)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-10)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 25.502777 atol=1e-4 rtol=1e-4

    # sparse methods
    (c, A, b, G, h, cones, cone_idxs) = build_envelope(2, 5, 1, 5, use_data = true, dense = false)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-10)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 25.502777 atol=1e-4 rtol=1e-4
end

function envelope2(; verbose::Bool = true)
    # dense methods
    (c, A, b, G, h, cones, cone_idxs) = build_envelope(2, 4, 2, 7, dense = true)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    rd = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test rd.status == :Optimal

    # sparse methods
    (c, A, b, G, h, cones, cone_idxs) = build_envelope(2, 4, 2, 7, dense = false)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    rs = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test rs.status == :Optimal

    @test rs.primal_obj ≈ rd.primal_obj atol=1e-4 rtol=1e-4
end

function envelope3(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_envelope(2, 3, 3, 5, dense = false)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
end

function envelope4(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_envelope(2, 2, 4, 3, dense = false)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
end

function linearopt1(; verbose::Bool = true)
    # dense methods
    (c, A, b, G, h, cones, cone_idxs) = build_linearopt(25, 50, dense = true, tosparse = false)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    rd = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test rd.status == :Optimal

    # sparse methods
    (c, A, b, G, h, cones, cone_idxs) = build_linearopt(25, 50, dense = true, tosparse = true)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    rs = solveandcheck(model, SO.HSDSolver(model, verbose = verbose))
    @test rs.status == :Optimal

    @test rs.primal_obj ≈ rd.primal_obj atol=1e-4 rtol=1e-4
end

function linearopt2(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_linearopt(500, 1000, use_data = true, dense = true)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test r.primal_obj ≈ 2055.807 atol=1e-4 rtol=1e-4
end

# for namedpoly tests, most optimal values are taken from https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html

function namedpoly1(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:butcher, 2)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 1.4393333333 atol=1e-4 rtol=1e-4
end

function namedpoly2(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:caprasse, 4)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 3.1800966258 atol=1e-4 rtol=1e-4
end

function namedpoly3(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:goldsteinprice, 6)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-9, tol_rel_opt = 1e-8, tol_abs_opt = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 3 atol=1e-4 rtol=1e-4
end

function namedpoly4(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:heart, 2)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 1.36775 atol=1e-4 rtol=1e-4
end

function namedpoly5(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:lotkavolterra, 3)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 20.8 atol=1e-4 rtol=1e-4
end

function namedpoly6(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:magnetism7, 2)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 0.25 atol=1e-4 rtol=1e-4
end

function namedpoly7(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:motzkin, 7)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 0 atol=1e-4 rtol=1e-4
end

function namedpoly8(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:reactiondiffusion, 4)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 36.71269068 atol=1e-4 rtol=1e-4
end

function namedpoly9(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:robinson, 8)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 0.814814 atol=1e-4 rtol=1e-4
end

function namedpoly10(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:rosenbrock, 5)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-10, tol_rel_opt = 1e-9, tol_abs_opt = 1e-9)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 0 atol=1e-3 rtol=1e-3
end

function namedpoly11(; verbose::Bool = true)
    (c, A, b, G, h, cones, cone_idxs) = build_namedpoly(:schwefel, 4)
    model = MO.Linear(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = verbose, tol_feas = 1e-8)
    r = solveandcheck(model, solver)
    @test r.status == :Optimal
    @test abs(r.primal_obj) ≈ 0 atol=1e-3 rtol=1e-3
end
