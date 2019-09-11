#=
Copyright 2018, Chris Coey and contributors

naive+elimination linear system solver
solves linear system in naive.jl via the following procedure

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

TODO add iterative method
=#

mutable struct NaiveElimSystemSolver{T <: Real} <: SystemSolver{T}
    use_sparse::Bool

    solver::Solver{T}

    lhs_copy
    lhs
    rhs::Matrix{T}

    x1
    x2
    y1
    y2
    z1
    z2
    z1_k
    z2_k
    s1::Vector{T}
    s2::Vector{T}
    s1_k
    s2_k

    function NaiveElimSystemSolver{T}(; use_sparse::Bool = false) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_sparse = use_sparse
        return system_solver
    end
end

function load(system_solver::NaiveElimSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.solver = solver

    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    npq1 = n + p + q + 1

    if system_solver.use_sparse
        system_solver.lhs_copy = T[
            spzeros(T,n,n)  model.A'        model.G'              model.c;
            -model.A        spzeros(T,p,p)  spzeros(T,p,q)        model.b;
            -model.G        spzeros(T,q,p)  sparse(one(T)*I,q,q)  model.h;
            -model.c'       -model.b'       -model.h'             one(T);
        ]
        @assert issparse(system_solver.lhs_copy)
    else
        system_solver.lhs_copy = T[
            zeros(T,n,n)  model.A'      model.G'              model.c;
            -model.A      zeros(T,p,p)  zeros(T,p,q)          model.b;
            -model.G      zeros(T,q,p)  Matrix(one(T)*I,q,q)  model.h;
            -model.c'     -model.b'     -model.h'             one(T);
        ]
    end

    system_solver.lhs = similar(system_solver.lhs_copy)

    rhs = Matrix{T}(undef, npq1, 2)
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

function get_combined_directions(system_solver::NaiveElimSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
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

    # update rhs matrix
    @timeit solver.timer "rhs update" begin
    x1 .= solver.x_residual
    x2 .= zero(T)
    y1 .= solver.y_residual
    y2 .= zero(T)
    sqrtmu = sqrt(mu)
    for k in eachindex(cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cones[k])
        @. s1_k[k] = -duals_k
        @. s2_k[k] = -duals_k - grad_k * sqrtmu
    end

    tau_rhs1 = -kap
    tau_rhs2 = -kap + mu / tau
    kap_rhs1 = kap + solver.primal_obj_t - solver.dual_obj_t

    # update lhs matrix
    @timeit solver.timer "lhs update" begin
    copyto!(lhs, system_solver.lhs_copy)
    mtt = mu / tau / tau
    lhs[end, end] = mtt

    for k in eachindex(cones)
        cone_k = cones[k]
        idxs = model.cone_idxs[k]
        rows = (n + p) .+ idxs
        H = Cones.hess(cone_k)
        if Cones.use_dual(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            @views copyto!(lhs[rows, rows], H)
            @views copyto!(z1_k[k], solver.z_residual[idxs])
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @timeit solver.timer "mul1" @views mul!(lhs[rows, 1:n], H, model.G[idxs, :], -one(T), false)
            @timeit solver.timer "mul2" @views mul!(lhs[rows, end], H, model.h[idxs])
            @timeit solver.timer "mul3" @views mul!(z1_k[k], H, solver.z_residual[idxs])
        end
    end
    z1 .+= s1
    z2 .= s2
    # -c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs
    rhs[end, 1] = kap_rhs1 + tau_rhs1
    rhs[end, 2] = tau_rhs2

    # solve system
    if system_solver.use_sparse
        @timeit solver.timer "division" rhs .= lu(lhs) \ rhs
    else
        @timeit solver.timer "division" ldiv!(lu!(lhs), rhs)
    end

    # lift to get s and kap
    tau1 = rhs[end, 1]
    tau2 = rhs[end, 2]

    # s = -G*x + h*tau - zrhs
    @. s1 = model.h * tau1 - solver.z_residual
    @timeit solver.timer "mul4" mul!(s1, model.G, x1, -one(T), true)
    s2 .= model.h
    @timeit solver.timer "mul5" mul!(s2, model.G, x2, -one(T), tau2)

    # kap = taurhs - mu/(taubar^2)*tau
    kap1 = tau_rhs1 - mtt * tau1
    kap2 = tau_rhs2 - mtt * tau2

    return (x1, x2, y1, y2, z1, z2, s1, s2, tau1, tau2, kap1, kap2)
end
