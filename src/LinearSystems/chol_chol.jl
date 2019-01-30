#=
Copyright 2018, Chris Coey and contributors

performs one 3x3 linear system solve
reduces 3x3 system to a 2x2 system and solves via two sequential (dense) choleskys (see CVXOPT)
if there are no equality constraints, only one cholesky is needed

TODO are there cases where a sparse cholesky would perform better?
=#

mutable struct CholChol <: LinearSystemSolver
    model::Models.Model

    function CholChol(model::Models.Model)
        linear_solver = new()
        linear_solver.model = model
        return linear_solver
    end
end

# solve 3x3 system for x, y, z
function solve_linear(
    x_rhs::Vector{Float64},
    y_rhs::Vector{Float64},
    z_rhs::Vector{Float64},
    mu::Float64,
    linear_solver::CholChol,
    )
    model = linear_solver.model
    (n, p, q, P, A, G, cones, cone_idxs) = (model.n, model.p, model.q, model.P, model.A, model.G, model.cones, model.cone_idxs)

    HG = Matrix{Float64}(undef, q, n)
    for k in eachindex(cone.cones)
        Gview = view(G, cone.idxs[k], :)
        HGview = view(HG, cone.idxs[k], :)
        if cone.cones[k].use_dual
            calcHiarr!(HGview, Gview, cone.cones[k])
            HGview ./= mu
        else
            calcHarr!(HGview, Gview, cone.cones[k])
            HGview .*= mu
        end
    end
    GHG = Symmetric(G' * HG)
    PGHG = Symmetric(P + GHG)
    # F1 = cholesky!(PGHG, Val(true), check = false)
    F1 = cholesky(PGHG, check = false) # TODO allow pivot
    singular = !isposdef(F1)

    if singular
        println("singular PGHG")
        PGHGAA = Symmetric(P + GHG + A' * A)
        # F1 = cholesky!(PGHGAA, Val(true), check = false)
        F1 = cholesky(PGHGAA, check = false) # TODO allow pivot
        if !isposdef(F1)
            error("could not fix singular PGHG")
        end
    end

    # LA = A'[F1.p, :]
    LA = A'
    # ldiv!(F1.L, LA)
    LA = F1.L \ LA
    ALLA = Symmetric(LA' * LA)
    # F2 = cholesky!(ALLA, Val(fal), check = false) # TODO avoid if no equalities?
    F2 = cholesky(ALLA, check = false) # TODO allow pivot; TODO avoid if no equalities?
    if !isposdef(F2)
        error("singular ALLA")
    end

    Hz = similar(z_rhs3)
    for k in eachindex(cone.cones)
        zview = view(z_rhs3, cone.idxs[k], :)
        Hzview = view(Hz, cone.idxs[k], :)
        if cone.cones[k].use_dual
            calcHiarr!(Hzview, zview, cone.cones[k])
            Hzview ./= mu
        else
            calcHarr!(Hzview, zview, cone.cones[k])
            Hzview .*= mu
        end
    end
    xGHz = x_rhs + G' * Hz
    if singular
        xGHz += A' * y_rhs # TODO should this be minus
    end

    # LxGHz = xGHz[F1.p]
    # LxGHz = copy(xGHz)
    # ldiv!(F1.L, LxGHz)
    LxGHz = F1.L \ xGHz

    y_sol = LA' * LxGHz - y_rhs
    # ldiv!(F2, y)
    y_sol = F2 \ y_sol

    x_sol = xGHz - A' * y_sol
    # ldiv!(F1, x)
    x_sol = F1 \ x_sol

    z_sol = similar(z_rhs3)
    Gxz = G * x_sol - z_rhs3
    for k in eachindex(cone.cones)
        Gxzview = view(Gxz, cone.idxs[k], :)
        zview = view(z_sol, cone.idxs[k], :)
        if cone.cones[k].use_dual
            calcHiarr!(zview, Gxzview, cone.cones[k])
            zview ./= mu
        else
            calcHarr!(zview, Gxzview, cone.cones[k])
            zview .*= mu
        end
    end

    return (x_sol, y_sol, z_sol)
end
