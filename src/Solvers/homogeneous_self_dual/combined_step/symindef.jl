#=
Copyright 2018, Chris Coey and contributors

symmetric-indefinite linear system solver
does not require inverse hessian products

A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
(pr bar) -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
(du bar) -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs

eliminate tau (see CVXOPT paper)

symmetrize the LHS matrix by multiplying some equations by -1 and by premultiplying the z variable by (mu*H_k)^-1 for k using primal barrier

TODO reduce allocations
=#

mutable struct SymIndefCombinedHSDSystemSolver{T <: HypReal} <: CombinedHSDSystemSolver{T}
    use_sparse::Bool

    lhs_copy
    lhs
    rhs::Matrix{T}

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
    s1::Vector{T}
    s2::Vector{T}
    s3::Vector{T}
    s1_k
    s2_k
    s3_k

    function SymIndefCombinedHSDSystemSolver{T}(model::Models.LinearModel{T}; use_sparse::Bool = false) where {T <: HypReal}
        (n, p, q) = (model.n, model.p, model.q)
        npq = n + p + q
        system_solver = new{T}()
        system_solver.use_sparse = use_sparse

        # x y z
        # lower symmetric
        if use_sparse
            system_solver.lhs_copy = T[
                spzeros(T,n,n)  spzeros(T,n,p)  spzeros(T,n,q);
                model.A         spzeros(T,p,p)  spzeros(T,p,q);
                model.G         spzeros(T,q,p)  sparse(-one(T)*I,q,q);
            ]
            @assert issparse(system_solver.lhs_copy)
        else
            system_solver.lhs_copy = T[
                zeros(T,n,n)  zeros(T,n,p)  zeros(T,n,q);
                model.A       zeros(T,p,p)  zeros(T,p,q);
                model.G       zeros(T,q,p)  Matrix(-one(T)I,q,q);
            ]
        end

        system_solver.lhs = similar(system_solver.lhs_copy)
        # function view_k(k::Int)
        #     rows = (n + p) .+ model.cone_idxs[k]
        #     cols = Cones.use_dual(model.cones[k]) ? rows : (q + 1) .+ rows
        #     return view(system_solver.lhs, rows, cols)
        # end
        # system_solver.lhs_H_k = [view_k(k) for k in eachindex(model.cones)]

        rhs = Matrix{T}(undef, npq, 3)
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

        return system_solver
    end
end

function get_combined_directions(solver::HSDSolver{T}, system_solver::SymIndefCombinedHSDSystemSolver{T}) where {T <: HypReal}
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

    @. x1 = -model.c
    x2 .= solver.x_residual
    x3 .= zero(T)
    y1 .= model.b
    @. y2 = -solver.y_residual
    y3 .= zero(T)
    z1 .= model.h
    @. z2 .= -solver.z_residual
    z3 .= zero(T)

    # update lhs matrix
    copyto!(lhs, system_solver.lhs_copy)
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
    if system_solver.use_sparse
        F = ldlt(Symmetric(lhs, :L), check = false)
        if !issuccess(F)
            F = ldlt(Symmetric(lhs, :L), shift = 1e-6, check = true)
        end
        rhs .= F \ rhs
    else
        F = bunchkaufman!(Symmetric(lhs, :L), true, check = true) # TODO doesn't work for generic reals (need LDLT)
        ldiv!(F, rhs)
    end

    for k in eachindex(cones)
        if !Cones.use_dual(cones[k])
            # z_k is premultiplied by (mu * H_k)^-1
            H = Cones.hess(cones[k])
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
    (tau_corr, kap_corr) = lift!(x3, y3, z3, s3, zero(T), -kap + mu / tau)

    return (x2, x3, y2, y3, z2, z3, s2, s3, tau_pred, tau_corr, kap_pred, kap_corr)
end
