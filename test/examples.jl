#=
Copyright 2018, Chris Coey and contributors
=#

function _envelope1(; verbose, lscachetype)
    # dense methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 5, 1, 5, use_data=true, dense=true)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.pobj ≈ 25.502777 atol=1e-4 rtol=1e-4
    @test r.niters <= 35

    # sparse methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 5, 1, 5, use_data=true, dense=false)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.pobj ≈ 25.502777 atol=1e-4 rtol=1e-4
    @test r.niters <= 35
end

function _envelope2(; verbose, lscachetype)
    # dense methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 4, 2, 7, dense=true)
    rd = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test rd.status == :Optimal
    @test rd.niters <= 60

    # sparse methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 4, 2, 7, dense=false)
    rs = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test rs.status == :Optimal
    @test rs.niters <= 60

    @test rs.pobj ≈ rd.pobj atol=1e-4 rtol=1e-4
end

function _envelope3(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose, tolfeas=5e-7)
    (c, A, b, G, h, cone) = build_envelope(2, 3, 3, 5, dense=false)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 60
end

function _envelope4(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_envelope(2, 2, 4, 3, dense=false)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 55
end

function _linearopt1(; verbose, lscachetype)
    # dense methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_linearopt(25, 50, dense=true, tosparse=false)
    rd = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test rd.status == :Optimal
    @test rd.niters <= 35

    # sparse methods
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_linearopt(25, 50, dense=true, tosparse=true)
    rs = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test rs.status == :Optimal
    @test rs.niters <= 35

    @test rs.pobj ≈ rd.pobj atol=1e-4 rtol=1e-4
end

function _linearopt2(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose, tolfeas=1e-9)
    (c, A, b, G, h, cone) = build_linearopt(500, 1000, use_data=true, dense=true)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 70
    @test r.pobj ≈ 2055.807 atol=1e-4 rtol=1e-4
end

# for namedpoly tests, most optimal values are taken from https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html

function _namedpoly1(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:butcher, 2)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 45
    @test r.pobj ≈ -1.4393333333 atol=1e-4 rtol=1e-4
end

function _namedpoly2(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:caprasse, 4)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 45
    @test r.pobj ≈ -3.1800966258 atol=1e-4 rtol=1e-4
end

function _namedpoly3(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose, tolfeas=1e-9)
    (c, A, b, G, h, cone) = build_namedpoly(:goldsteinprice, 6)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype, atol=2e-3)
    @test r.status == :Optimal
    @test r.niters <= 65
    @test r.pobj ≈ 3 atol=1e-4 rtol=1e-4
end

function _namedpoly4(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:heart, 2)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    # @test r.niters <= 40
    @test r.pobj ≈ -1.36775 atol=1e-4 rtol=1e-4
end

function _namedpoly5(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:lotkavolterra, 3)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 40
    @test r.pobj ≈ -20.8 atol=1e-4 rtol=1e-4
end

function _namedpoly6(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:magnetism7, 2)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 35
    @test r.pobj ≈ -0.25 atol=1e-4 rtol=1e-4
end

function _namedpoly7(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:motzkin, 7)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 45
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
end

function _namedpoly8(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:reactiondiffusion, 4)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 40
    @test r.pobj ≈ -36.71269068 atol=1e-4 rtol=1e-4
end

function _namedpoly9(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose)
    (c, A, b, G, h, cone) = build_namedpoly(:robinson, 8)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 40
    @test r.pobj ≈ 0.814814 atol=1e-4 rtol=1e-4
end

function _namedpoly10(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose, tolfeas=1e-9)
    (c, A, b, G, h, cone) = build_namedpoly(:rosenbrock, 5)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype, atol=2e-3)
    @test r.status == :Optimal
    @test r.niters <= 65
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
end

function _namedpoly11(; verbose, lscachetype)
    mdl = Hypatia.Model(verbose=verbose, tolfeas=1e-9)
    (c, A, b, G, h, cone) = build_namedpoly(:schwefel, 4)
    r = solveandcheck(mdl, c, A, b, G, h, cone, lscachetype)
    @test r.status == :Optimal
    @test r.niters <= 60
    @test r.pobj ≈ 0 atol=1e-4 rtol=1e-4
end

function _namedpoly1_JuMP()
    # the Heart polynomial in a box
    (x, f, dom, truemin) = getpolydata(:heart)

    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=2, pts_factor=3)
    JuMP.optimize!(mdl)
    term_status = JuMP.termination_status(mdl)
    pobj = JuMP.objective_value(mdl)
    dobj = JuMP.objective_bound(mdl)
    pr_status = JuMP.primal_status(mdl)
    du_status = JuMP.dual_status(mdl)
    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4
end

function _namedpoly2_JuMP()
    # the Schwefel polynomial in a box
    (x, f, dom, truemin) = getpolydata(:schwefel)

    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=2, pts_factor=2*length(x))
    JuMP.optimize!(mdl)
    term_status = JuMP.termination_status(mdl)
    pobj = JuMP.objective_value(mdl)
    dobj = JuMP.objective_bound(mdl)
    pr_status = JuMP.primal_status(mdl)
    du_status = JuMP.dual_status(mdl)
    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4

    # SDP formulation
    mdl = build_JuMP_namedpoly_SDP(x, f, dom, d=2)
    JuMP.optimize!(mdl)
    term_status = JuMP.termination_status(mdl)
    pobj = JuMP.objective_value(mdl)
    dobj = JuMP.objective_bound(mdl)
    pr_status = JuMP.primal_status(mdl)
    du_status = JuMP.dual_status(mdl)
    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4
end

function _namedpoly3_JuMP()
    # the Magnetism polynomial in a ball
    (x, f, dom, truemin) = getpolydata(:magnetism7_ball)

    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=2, pts_factor=length(x))
    JuMP.optimize!(mdl)
    term_status = JuMP.termination_status(mdl)
    pobj = JuMP.objective_value(mdl)
    dobj = JuMP.objective_bound(mdl)
    pr_status = JuMP.primal_status(mdl)
    du_status = JuMP.dual_status(mdl)
    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4

    # SDP formulation
    mdl = build_JuMP_namedpoly_SDP(x, f, dom, d=2)
    JuMP.optimize!(mdl)
    term_status = JuMP.termination_status(mdl)
    pobj = JuMP.objective_value(mdl)
    dobj = JuMP.objective_bound(mdl)
    pr_status = JuMP.primal_status(mdl)
    du_status = JuMP.dual_status(mdl)
    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4
end

function _namedpoly4_JuMP()
    # the Motzkin polynomial in an ellipsoid containing two local minima in opposite orthants
    (x, f, dom, truemin) = getpolydata(:motzkin_ellipsoid)

    # WSOS formulation
    mdl = build_JuMP_namedpoly_WSOS(x, f, dom, d=7, pts_factor=4*length(x))
    JuMP.optimize!(mdl)
    term_status = JuMP.termination_status(mdl)
    pobj = JuMP.objective_value(mdl)
    dobj = JuMP.objective_bound(mdl)
    pr_status = JuMP.primal_status(mdl)
    du_status = JuMP.dual_status(mdl)
    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4

    # SDP formulation
    mdl = build_JuMP_namedpoly_SDP(x, f, dom, d=7)
    JuMP.optimize!(mdl)
    term_status = JuMP.termination_status(mdl)
    pobj = JuMP.objective_value(mdl)
    dobj = JuMP.objective_bound(mdl)
    pr_status = JuMP.primal_status(mdl)
    du_status = JuMP.dual_status(mdl)
    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4
end
