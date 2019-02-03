#=
Copyright 2018, Chris Coey and contributors

performs one 3x3 linear system solve
reduces 3x3 system to a 2x2 system and solves via two sequential (dense) choleskys (see CVXOPT)
if there are no equality constraints, only one cholesky is needed

TODO are there cases where a sparse cholesky would perform better?
=#

mutable struct CholChol <: LinearSystemSolver

    function CholChol(model::Models.Model)
        linear_solver = new()
        return linear_solver
    end
end

# solve 3x3 system for x, y, z
function solve_linear_system(
    x_rhs::Matrix{Float64},
    y_rhs::Matrix{Float64},
    z_rhs::Matrix{Float64},
    mu::Float64,
    model::Models.Linear,
    linear_solver::CholChol,
    )
    (n, p, q, A, G, cones, cone_idxs) = (model.n, model.p, model.q, model.A, model.G, model.cones, model.cone_idxs)

    HG = Matrix{Float64}(undef, q, n)
    for k in eachindex(cones)
        Gview = view(G, cone_idxs[k], :)
        if cones[k].use_dual
            HG[cone_idxs[k], :] = Cones.inv_hess(cones[k]) * (Gview ./ mu)
        else
            HG[cone_idxs[k], :] = Cones.hess(cones[k]) * (Gview .* mu)
        end
    end
    GHG = Symmetric(G' * HG)
    # F1 = cholesky!(GHG, Val(true), check = false)
    F1 = cholesky(GHG, check = false) # TODO allow pivot
    singular = !isposdef(F1)

    if singular
        println("singular GHG")
        GHGAA = Symmetric(GHG + A' * A)
        # F1 = cholesky!(GHGAA, Val(true), check = false)
        F1 = cholesky(GHGAA, check = false) # TODO allow pivot
        if !isposdef(F1)
            error("could not fix singular GHG")
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

    Hz = similar(z_rhs)
    for k in eachindex(cones)
        zview = view(z_rhs, cone_idxs[k], :)
        if cones[k].use_dual
            Hz[cone_idxs[k], :] = Cones.inv_hess(cones[k]) * (zview ./ mu)
        else
            Hz[cone_idxs[k], :] = Cones.hess(cones[k]) * (zview .* mu)
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

    z_sol = similar(z_rhs)
    Gxz = G * x_sol - z_rhs
    for k in eachindex(cones)
        Gxzview = view(Gxz, cone_idxs[k], :)
        if cones[k].use_dual
            z_sol[cone_idxs[k], :] = Cones.inv_hess(cones[k]) * (Gxzview ./ mu)
        else
            z_sol[cone_idxs[k], :] = Cones.hess(cones[k]) * (Gxzview .* mu)
        end
    end

    return (x_sol, y_sol, z_sol)
end
