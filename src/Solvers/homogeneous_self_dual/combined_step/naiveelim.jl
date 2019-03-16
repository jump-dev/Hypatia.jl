
#=
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
-G*x - s + h*tau = zrhs
-c'*x - b'*y - h'*z - kap = kaprhs
(pr bar) z_k + mu*H_k*s_k = srhs_k
(du bar) mu*H_k*z_k + s_k = srhs_k
kap + mu/(taubar^2)*tau = taurhs

eliminate s
-G*x - s + h*tau = zrhs
so if using primal barrier
z_k + mu*H_k*s_k = srhs_k --> s_k = (mu*H_k)\(srhs_k - z_k)
-->
-G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k
-->
-mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
or if using dual barrier
mu*H_k*z_k + s_k = srhs_k --> s_k = srhs_k - mu*H_k*z_k
-->
-G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k

eliminate kap
-c'x - b'y - h'z - kap = kaprhs
so
kap + mu/(taubar^2)*tau = taurhs --> kap = taurhs - mu/(taubar^2)*tau
-->
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs
=#


# TODO eliminate allocations


mutable struct NaiveElimCombinedHSDSystemSolver <: CombinedHSDSystemSolver
    lhs::Matrix{Float64}
    # lhs_H_k
    rhs::Matrix{Float64}
    x1
    x2
    y1
    y2
    z1
    z2
    z1_k
    z2_k
    s1::Vector{Float64}
    s2::Vector{Float64}
    s1_k
    s2_k

    function NaiveElimCombinedHSDSystemSolver(model::Models.LinearModel)
        (n, p, q) = (model.n, model.p, model.q)
        npq1 = n + p + q + 1
        system_solver = new()

        # TODO allow sparse lhs?
        system_solver.lhs = Matrix{Float64}(undef, npq1, npq1)
        # function view_k(k::Int)
        #     rows = (n + p) .+ model.cone_idxs[k]
        #     cols = Cones.use_dual(model.cones[k]) ? rows : (q + 1) .+ rows
        #     return view(system_solver.lhs, rows, cols)
        # end
        # system_solver.lhs_H_k = [view_k(k) for k in eachindex(model.cones)]

        rhs = similar(system_solver.lhs, npq1, 2)
        system_solver.rhs = rhs
        rows = 1:n
        system_solver.x1 = view(rhs, rows, 1)
        system_solver.x2 = view(rhs, rows, 2)
        rows = (n + 1):(n + p)
        system_solver.y1 = view(rhs, rows, 1)
        system_solver.y2 = view(rhs, rows, 2)
        rows = (n + p + 1):(n + p + q)
        system_solver.z1 = view(rhs, rows, 1)
        system_solver.z2 = view(rhs, rows, 2)
        z_start = n + p
        system_solver.z1_k = [view(rhs, z_start .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
        system_solver.z2_k = [view(rhs, z_start .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]
        system_solver.s1 = similar(rhs, q)
        system_solver.s2 = similar(rhs, q)
        system_solver.s1_k = [view(system_solver.s1, model.cone_idxs[k]) for k in eachindex(model.cones)]
        system_solver.s2_k = [view(system_solver.s2, model.cone_idxs[k]) for k in eachindex(model.cones)]

        return system_solver
    end
end

function get_combined_directions(solver::HSDSolver, system_solver::NaiveElimCombinedHSDSystemSolver)
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
    y1 = system_solver.y1
    y2 = system_solver.y2
    z1 = system_solver.z1
    z2 = system_solver.z2
    z1_k = system_solver.z1_k
    z2_k = system_solver.z2_k
    s1 = system_solver.s1
    s2 = system_solver.s2
    s1_k = system_solver.s1_k
    s2_k = system_solver.s2_k

    x1 .= solver.x_residual
    x2 .= 0.0
    y1 .= solver.y_residual
    y2 .= 0.0
    z1 .= solver.z_residual
    z2 .= 0.0
    for k in eachindex(cones)
        duals_k = solver.point.dual_views[k]
        g = Cones.grad(cones[k])
        @. s1_k[k] = -duals_k
        @. s2_k[k] = -duals_k - mu * g
    end
    tau_rhs = [-kap, -kap + mu / tau]
    kap_rhs = [kap + solver.primal_obj_t - solver.dual_obj_t, 0.0]

    mtt = mu/tau/tau

    # x y z tau
    lhs .= [
        zeros(n,n)  model.A'    model.G'          model.c;
        -model.A    zeros(p,p)  zeros(p,q)        model.b;
        -model.G    zeros(q,p)  Matrix(1.0I,q,q)  model.h;
        -model.c'   -model.b'   -model.h'         mtt;
    ]

    for k in eachindex(cones)
        cone_k = cones[k]
        idxs = model.cone_idxs[k]
        rows = (n + p) .+ idxs
        H = Cones.hess(cones[k])
        if Cones.use_dual(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            lhs[rows, rows] .= mu * H
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            lhs[rows, 1:n] .= -mu * H * model.G[idxs, :]
            lhs[rows, end] .= mu * H * model.h[idxs]
            z1_k[k] .= mu * H * z1_k[k]
            z2_k[k] .= mu * H * z2_k[k]
        end
    end
    z1 .+= s1
    z2 .+= s2
    # -c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs
    rhs[end, :] .= kap_rhs + tau_rhs

    # solve system
    ldiv!(lu!(lhs), rhs)

    # lift to get s and kap
    tau1 = rhs[end, 1]
    tau2 = rhs[end, 2]

    # s = -G*x + h*tau - zrhs
    s1 .= -model.G * x1 + model.h * tau1 - solver.z_residual
    s2 .= -model.G * x2 + model.h * tau2

    # kap = taurhs - mu/(taubar^2)*tau
    kap1 = tau_rhs[1] - mtt * tau1
    kap2 = tau_rhs[2] - mtt * tau2

    return (x1, x2, y1, y2, z1, z2, s1, s2, tau1, tau2, kap1, kap2)
end
