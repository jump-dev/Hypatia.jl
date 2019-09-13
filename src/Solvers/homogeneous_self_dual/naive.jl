#=
Copyright 2018, Chris Coey and contributors

naive linear system solver

A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
-G*x + h*tau - s = zrhs
-c'*x - b'*y - h'*z - kap = taurhs
(pr bar) z_k + mu*H_k*s_k = srhs_k
(du bar) mu*H_k*z_k + s_k = srhs_k
mu/(taubar^2)*tau + kap = kaprhs

TODO optimize iterative method
=#

mutable struct NaiveSystemSolver{T <: Real} <: SystemSolver{T}
    use_iterative::Bool
    use_sparse::Bool

    solver::Solver{T}

    x1
    x2
    y1
    y2
    z1
    z2
    tau_row::Int
    s1
    s2
    s1_k
    s2_k

    lhs_copy
    lhs
    lhs_H_k
    rhs::Matrix{T}
    prevsol1::Vector{T}
    prevsol2::Vector{T}

    solvesol
    solvecache

    function NaiveSystemSolver{T}(; use_iterative::Bool = false, use_sparse::Bool = false) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_iterative = use_iterative
        system_solver.use_sparse = use_sparse
        return system_solver
    end
end

function load(system_solver::NaiveSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.solver = solver

    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    dim = n + p + 2q + 2

    rhs = zeros(T, dim, 2)
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
    tau_row = n + p + q + 1
    system_solver.tau_row = tau_row
    rows = tau_row .+ (1:q)
    system_solver.s1 = view(rhs, rows, 1)
    system_solver.s2 = view(rhs, rows, 2)
    system_solver.s1_k = [view(rhs, tau_row .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
    system_solver.s2_k = [view(rhs, tau_row .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]

    # x y z kap s tau
    if system_solver.use_iterative
        system_solver.prevsol1 = zeros(T, dim)
        system_solver.prevsol2 = zeros(T, dim)
        # TODO compare approaches
        # lhs_lin_map(arr) = @views vcat(apply_lhs(solver, arr[1:n], arr[(n + 1):(n + p)], arr[(n + p + 1):(n + p + q)], arr[tau_row], arr[tau_row .+ (1:q)], arr[end]))
        # system_solver.lhs_map = LinearMaps.FunctionMap{T}(lhs_lin_map)
        system_solver.lhs = setup_block_lhs(solver)
    else
        if system_solver.use_sparse
            system_solver.lhs_copy = T[
                spzeros(T,n,n)  model.A'        model.G'              model.c       spzeros(T,n,q)         spzeros(T,n);
                -model.A        spzeros(T,p,p)  spzeros(T,p,q)        model.b       spzeros(T,p,q)         spzeros(T,p);
                -model.G        spzeros(T,q,p)  spzeros(T,q,q)        model.h       sparse(-one(T)*I,q,q)  spzeros(T,q);
                -model.c'       -model.b'       -model.h'             zero(T)       spzeros(T,1,q)         -one(T);
                spzeros(T,q,n)  spzeros(T,q,p)  sparse(one(T)*I,q,q)  spzeros(T,q)  sparse(one(T)*I,q,q)   spzeros(T,q);
                spzeros(T,1,n)  spzeros(T,1,p)  spzeros(T,1,q)        one(T)        spzeros(T,1,q)         one(T);
                ]
            dropzeros!(system_solver.lhs_copy)
            @assert issparse(system_solver.lhs_copy)
        else
            system_solver.lhs_copy = T[
                zeros(T,n,n)  model.A'      model.G'              model.c     zeros(T,n,q)           zeros(T,n);
                -model.A      zeros(T,p,p)  zeros(T,p,q)          model.b     zeros(T,p,q)           zeros(T,p);
                -model.G      zeros(T,q,p)  zeros(T,q,q)          model.h     Matrix(-one(T)*I,q,q)  zeros(T,q);
                -model.c'     -model.b'     -model.h'             zero(T)     zeros(T,1,q)           -one(T);
                zeros(T,q,n)  zeros(T,q,p)  Matrix(one(T)*I,q,q)  zeros(T,q)  Matrix(one(T)*I,q,q)   zeros(T,q);
                zeros(T,1,n)  zeros(T,1,p)  zeros(T,1,q)          one(T)      zeros(T,1,q)           one(T);
                ]
        end

        system_solver.lhs = similar(system_solver.lhs_copy)
        function view_k(k::Int)
            rows = tau_row .+ model.cone_idxs[k]
            if Cones.use_dual(model.cones[k])
                cols = (n + p) .+ model.cone_idxs[k]
            else
                cols = rows
            end
            return view(system_solver.lhs, rows, cols)
        end
        system_solver.lhs_H_k = [view_k(k) for k in eachindex(model.cones)]

        if !system_solver.use_sparse
            system_solver.solvesol = Matrix{T}(undef, size(system_solver.lhs, 1), 2)
            system_solver.solvecache = HypLUSolveCache(system_solver.solvesol, system_solver.lhs, rhs)
        end
    end

    return system_solver
end

function get_combined_directions(system_solver::NaiveSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    cones = model.cones
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    tau_row = system_solver.tau_row
    x1 = system_solver.x1
    x2 = system_solver.x2
    y1 = system_solver.y1
    y2 = system_solver.y2
    z1 = system_solver.z1
    z2 = system_solver.z2
    s1 = system_solver.s1
    s2 = system_solver.s2
    s1_k = system_solver.s1_k
    s2_k = system_solver.s2_k

    sqrtmu = sqrt(solver.mu)
    mtt = solver.mu / solver.tau / solver.tau

    # update rhs matrix
    x1 .= solver.x_residual
    x2 .= zero(T)
    y1 .= solver.y_residual
    y2 .= zero(T)
    z1 .= solver.z_residual
    z2 .= zero(T)
    rhs[tau_row, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    rhs[tau_row, 2] = zero(T)
    for k in eachindex(cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cones[k])
        @. s1_k[k] = -duals_k
        @. s2_k[k] = -duals_k - grad_k * sqrtmu
    end
    rhs[end, 1] = -solver.kap
    rhs[end, 2] = -solver.kap + solver.mu / solver.tau

    # solve system
    if system_solver.use_iterative
        # TODO need preconditioner
        # TODO optimize for this case, including applying the blocks of the LHS
        # TODO pick which square non-symm method to use
        # TODO prealloc whatever is needed inside the solver
        # TODO possibly fix IterativeSolvers so that methods can take matrix RHS, however the two columns may take different number of iters needed to converge
        lhs.blocks[end - 1][1] = mtt
        dim = size(lhs, 2)

        rhs1 = view(rhs, :, 1)
        IterativeSolvers.gmres!(system_solver.prevsol1, lhs, rhs1, restart = dim)
        copyto!(rhs1, system_solver.prevsol1)

        rhs2 = view(rhs, :, 2)
        IterativeSolvers.gmres!(system_solver.prevsol2, lhs, rhs2, restart = dim)
        copyto!(rhs2, system_solver.prevsol2)
    else
        # update lhs matrix
        copyto!(lhs, system_solver.lhs_copy)
        lhs[end, tau_row] = mtt
        for k in eachindex(cones)
            copyto!(system_solver.lhs_H_k[k], Cones.hess(cones[k]))
        end

        if system_solver.use_sparse
            rhs .= lu(lhs) \ rhs
        else
            if !hyp_lu_solve!(system_solver.solvecache, system_solver.solvesol, lhs, rhs)
                @warn("numerical failure: could not fix linear solve failure (mu is $mu)")
            end
            copyto!(rhs, system_solver.solvesol)
        end
    end

    return (x1, x2, y1, y2, z1, z2, rhs[tau_row, 1], rhs[tau_row, 2], s1, s2, rhs[end, 1], rhs[end, 2])
end

# block matrix for efficient multiplication
function setup_block_lhs(solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = n + p + q + 1
    rc1 = 1:n
    rc2 = n .+ (1:p)
    rc3 = (n + p) .+ (1:q)
    rc4 = tau_row:tau_row
    rc5 = tau_row .+ (1:q)
    dim = tau_row + q + 1
    rc6 = dim:dim

    k_len = 2 * length(model.cones)
    cone_rows = Vector{UnitRange{Int}}(undef, k_len)
    cone_cols = Vector{UnitRange{Int}}(undef, k_len)
    cone_blocks = Vector{Any}(undef, k_len)
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        rows = tau_row .+ idxs_k
        k1 = 2k - 1
        k2 = 2k
        cone_rows[k1] = cone_rows[k2] = rows
        cone_cols[k1] = (n + p) .+ idxs_k
        cone_cols[k2] = rows
        if Cones.use_dual(cone_k)
            cone_blocks[k1] = cone_k
            cone_blocks[k2] = I
        else
            cone_blocks[k1] = I
            cone_blocks[k2] = cone_k
        end
    end

    block_lhs = BlockMatrix{T}(dim, dim,
        [cone_blocks...,
            model.A', model.G', reshape(model.c, :, 1),
            -model.A, reshape(model.b, :, 1),
            -model.G, reshape(model.h, :, 1), -I,
            -model.c', -model.b', -model.h', -ones(T, 1, 1),
            ones(T, 1, 1), ones(T, 1, 1)],
        [cone_rows...,
            rc1, rc1, rc1,
            rc2, rc2,
            rc3, rc3, rc3,
            rc4, rc4, rc4, rc4,
            rc6, rc6],
        [cone_cols...,
            rc2, rc3, rc4,
            rc1, rc4,
            rc1, rc4, rc5,
            rc1, rc2, rc3, rc6,
            rc4, rc6],
        )

    return block_lhs
end

function apply_lhs(solver, x_in, y_in, z_in, tau_in, s_in, kap_in)
    model = solver.model

    # A'*y + G'*z + c*tau = xrhs
    x_out = model.A' * y_in + model.G' * z_in + model.c * tau_in
    # -A*x + b*tau = yrhs
    y_out = -model.A * x_in + model.b * tau_in
    # -G*x + h*tau - s = zrhs
    z_out = -model.G * x_in + model.h * tau_in - s_in
    # -c'*x - b'*y - h'*z - kap = taurhs
    tau_out = -dot(model.c, x_in) - dot(model.b, y_in) - dot(model.h, z_in) - kap_in

    s_out = similar(z_out)
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        if Cones.use_dual(cone_k)
            # (du bar) mu*H_k*z_k + s_k = srhs_k
            @views Cones.hess_prod!(s_out[idxs_k], z_in[idxs_k], cone_k)
            @views @. s_out[idxs_k] += s_in[idxs_k]
        else
            # (pr bar) z_k + mu*H_k*s_k = srhs_k
            @views Cones.hess_prod!(s_out[idxs_k], s_in[idxs_k], cone_k)
            @views @. s_out[idxs_k] += z_in[idxs_k]
        end
    end

    # mu/(taubar^2)*tau + kap = kaprhs
    kap_out = kap_in + solver.mu / solver.tau * tau_in / solver.tau

    return (x_out, y_out, z_out, tau_out, s_out, kap_out)
end
