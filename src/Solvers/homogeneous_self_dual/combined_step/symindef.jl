
#=
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
(pr bar) -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
(du bar) -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs

eliminate tau (see CVXOPT paper)

symmetrize the LHS matrix by multiplying some equations by -1 and by premultiplying the z variable by (mu*H_k)^-1 for k using primal barrier
=#


# TODO eliminate allocations


mutable struct SymIndefCombinedHSDSystemSolver <: CombinedHSDSystemSolver
    lhs::Matrix{Float64}
    # lhs_H_k
    rhs::Matrix{Float64}
    x1
    x2
    x3
    y1
    y2
    y3
    z1
    z2
    z3
    z1_k
    z2_k
    z3_k
    s1::Vector{Float64}
    s2::Vector{Float64}
    s3::Vector{Float64}
    s1_k
    s2_k
    s3_k

    function SymIndefCombinedHSDSystemSolver(model::Models.LinearModel)
        (n, p, q) = (model.n, model.p, model.q)
        npq = n + p + q
        system_solver = new()

        # TODO allow sparse lhs?
        system_solver.lhs = Matrix{Float64}(undef, npq, npq)
        # function view_k(k::Int)
        #     rows = (n + p) .+ model.cone_idxs[k]
        #     cols = Cones.use_dual(model.cones[k]) ? rows : (q + 1) .+ rows
        #     return view(system_solver.lhs, rows, cols)
        # end
        # system_solver.lhs_H_k = [view_k(k) for k in eachindex(model.cones)]

        rhs = similar(system_solver.lhs, npq, 3)
        system_solver.rhs = rhs
        rows = 1:n
        system_solver.x1 = view(rhs, rows, 1)
        system_solver.x2 = view(rhs, rows, 2)
        system_solver.x3 = view(rhs, rows, 3)
        rows = (n + 1):(n + p)
        system_solver.y1 = view(rhs, rows, 1)
        system_solver.y2 = view(rhs, rows, 2)
        system_solver.y3 = view(rhs, rows, 3)
        rows = (n + p + 1):(n + p + q)
        system_solver.z1 = view(rhs, rows, 1)
        system_solver.z2 = view(rhs, rows, 2)
        system_solver.z3 = view(rhs, rows, 3)
        z_start = n + p
        system_solver.z1_k = [view(rhs, z_start .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
        system_solver.z2_k = [view(rhs, z_start .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]
        system_solver.z3_k = [view(rhs, z_start .+ model.cone_idxs[k], 3) for k in eachindex(model.cones)]
        system_solver.s1 = similar(rhs, q)
        system_solver.s2 = similar(rhs, q)
        system_solver.s3 = similar(rhs, q)
        system_solver.s1_k = [view(system_solver.s1, model.cone_idxs[k]) for k in eachindex(model.cones)]
        system_solver.s2_k = [view(system_solver.s2, model.cone_idxs[k]) for k in eachindex(model.cones)]
        system_solver.s3_k = [view(system_solver.s3, model.cone_idxs[k]) for k in eachindex(model.cones)]

        return system_solver
    end
end

function get_combined_directions(solver::HSDSolver, system_solver::SymIndefCombinedHSDSystemSolver)
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    mu = solver.mu
    tau = solver.tau
    kap = solver.kap

    x1 = system_solver.x1
    x2 = system_solver.x2
    x3 = system_solver.x3
    y1 = system_solver.y1
    y2 = system_solver.y2
    y3 = system_solver.y3
    z1 = system_solver.z1
    z2 = system_solver.z2
    z3 = system_solver.z3
    z1_k = system_solver.z1_k
    z2_k = system_solver.z2_k
    z3_k = system_solver.z3_k
    s1 = system_solver.s1
    s2 = system_solver.s2
    s3 = system_solver.s3
    # s1_k = system_solver.s1_k
    # s2_k = system_solver.s2_k
    # s3_k = system_solver.s3_k

    @. x1 = -model.c
    x2 .= solver.x_residual
    x3 .= 0.0
    y1 .= model.b
    @. y2 = -solver.y_residual
    y3 .= 0.0
    z1 .= model.h
    @. z2 .= -solver.z_residual
    # z3 .= 0.0

    # x y z
    # lower symmetric
    lhs .= [
        zeros(n,n)  zeros(n,p)  zeros(n,q);
        model.A     zeros(p,p)  zeros(p,q);
        model.G     zeros(q,p)  Matrix(-1.0I,q,q);
    ]

    for k in eachindex(cones)
        cone_k = cones[k]
        idxs = model.cone_idxs[k]
        rows = (n + p) .+ idxs
        duals_k = solver.point.dual_views[k]
        H = Cones.hess(cone_k)
        g = Cones.grad(cone_k)
        if Cones.use_dual(cone_k)
            # G_k*x - mu*H_k*z_k = zrhs_k + srhs_k
            lhs[rows, rows] .= -mu * H
            @. z2_k[k] += duals_k
            @. z3_k[k] = duals_k + mu * g
        else
            # mu*H_k*G_k*x - z_k = mu*H_k*zrhs_k + srhs_k
            lhs[rows, 1:n] .= mu * H * model.G[idxs, :]
            z1_k[k] .= mu * H * z1_k[k]
            z2_k[k] .= mu * H * z2_k[k]
            @. z2_k[k] += duals_k
            @. z3_k[k] = duals_k + mu * g

            # symmetrize: z_k is premultiplied by (mu * H_k)^-1
            lhs[rows, rows] .= -mu * H
        end
    end

    # solve system
    ldiv!(bunchkaufman!(Symmetric(lhs, :L)), rhs)

    for k in eachindex(cones)
        H = Cones.hess(cones[k])
        if !Cones.use_dual(cones[k])
            # z_k is premultiplied by (mu * H_k)^-1
            z1_k[k] .= mu * H * z1_k[k]
            z2_k[k] .= mu * H * z2_k[k]
            z3_k[k] .= mu * H * z3_k[k]
        end
    end

    # lift to HSDE space
    tau_denom = mu / tau / tau - dot(model.c, x1) - dot(model.b, y1) - dot(model.h, z1)

    function lift!(x, y, z, s, tau_rhs, kap_rhs)
        tau_sol = (tau_rhs + kap_rhs + dot(model.c, x) + dot(model.b, y) + dot(model.h, z)) / tau_denom
        @. x += tau_sol * x1
        @. y += tau_sol * y1
        @. z += tau_sol * z1
        mul!(s, model.G, x)
        @. s = -s + tau_sol * model.h
        kap_sol = -dot(model.c, x) - dot(model.b, y) - dot(model.h, z) - tau_rhs
        return (tau_sol, kap_sol)
    end

    (tau_pred, kap_pred) = lift!(x2, y2, z2, s2, kap + solver.primal_obj_t - solver.dual_obj_t, -kap)
    @. s2 -= solver.z_residual
    (tau_corr, kap_corr) = lift!(x3, y3, z3, s3, 0.0, -kap + mu / tau)

    return (x2, x3, y2, y3, z2, z3, s2, s3, tau_pred, tau_corr, kap_pred, kap_corr)
end
