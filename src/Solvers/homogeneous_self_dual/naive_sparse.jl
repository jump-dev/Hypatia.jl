#=
Copyright 2018, Chris Coey and contributors

naive linear system solver

A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
-G*x - s + h*tau = zrhs
-c'*x - b'*y - h'*z - kap = kaprhs
(pr bar) z_k + mu*H_k*s_k = srhs_k
(du bar) mu*H_k*z_k + s_k = srhs_k
kap + mu/(taubar^2)*tau = taurhs

TODO optimize iterative method
=#

mutable struct NaiveSparseSystemSolver{T <: Real} <: SystemSolver{T}
    use_iterative::Bool

    solver::Solver{T}

    lhs_copy
    lhs
    lhs_H_k
    lhs_H_Vs
    H_start::Int
    Hinv_start::Int
    Is::Vector{Int}
    Js::Vector{Int}
    Vs::Vector{T}

    rhs::Matrix{T}
    prevsol1::Vector{T}
    prevsol2::Vector{T}

    x1
    x2
    y1
    y2
    z1
    z2
    z1_k
    z2_k
    s1
    s2
    kap_row::Int
    kap_idx::Int

    function NaiveSparseSystemSolver{T}(; use_iterative::Bool = false, use_sparse::Bool = false) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_iterative = use_iterative
        return system_solver
    end
end

function load(system_solver::NaiveSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.solver = solver

    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    dim = n + p + 2q + 2

    system_solver.rhs = zeros(T, dim, 2)
    rows = 1:n
    system_solver.x1 = view(system_solver.rhs, rows, 1)
    system_solver.x2 = view(system_solver.rhs, rows, 2)
    rows = (n + 1):(n + p)
    system_solver.y1 = view(system_solver.rhs, rows, 1)
    system_solver.y2 = view(system_solver.rhs, rows, 2)
    rows = (n + p + 1):(n + p + q)
    system_solver.z1 = view(system_solver.rhs, rows, 1)
    system_solver.z2 = view(system_solver.rhs, rows, 2)
    z_start = n + p
    system_solver.z1_k = [view(system_solver.rhs, z_start .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
    system_solver.z2_k = [view(system_solver.rhs, z_start .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]
    rows = (n + p + q + 2):(n + p + 2q + 1)
    system_solver.s1 = view(system_solver.rhs, rows, 1)
    system_solver.s2 = view(system_solver.rhs, rows, 2)
    system_solver.kap_row = n + p + q + 1


    lhs_actual = T[
        spzeros(T,n,n)  model.A'        model.G'              spzeros(T,n)  spzeros(T,n,q)         model.c;
        -model.A        spzeros(T,p,p)  spzeros(T,p,q)        spzeros(T,p)  spzeros(T,p,q)         model.b;
        spzeros(T,q,n)  spzeros(T,q,p)  sparse(one(T)*I,q,q)  spzeros(T,q)  sparse(one(T)*I,q,q)   spzeros(T,q);
        spzeros(T,1,n)  spzeros(T,1,p)  spzeros(T,1,q)        one(T)        spzeros(T,1,q)         one(T);
        -model.G        spzeros(T,q,p)  spzeros(T,q,q)        spzeros(T,q)  sparse(-one(T)*I,q,q)  model.h;
        -model.c'       -model.b'       -model.h'             -one(T)       spzeros(T,1,q)         zero(T);
        ]

    hess_nnzs = sum(Cones.dimension(cone_k) + Cones.dimension(cone_k) ^ 2 for cone_k in model.cones)

    total_nnz = 2 * (nnz(sparse(model.A)) + nnz(sparse(model.G)) + n + p + q + 1) + q + 1 + hess_nnzs
    Is = system_solver.Is = Vector{Int}(undef, total_nnz)
    Js = system_solver.Js = Vector{Int}(undef, total_nnz)
    Vs = system_solver.Vs = Vector{T}(undef, total_nnz)

    function add_I_J_V(k, start_row, start_col, vec::Vector{T})
        if !isempty(vec)
            for i in eachindex(vec)
                Is[k] = i + start_row
                Js[k] = start_col + 1
                Vs[k] = vec[i]
                k += 1
            end
        end
        return k
    end

    function add_I_J_V(k, start_row, start_col, vec::Adjoint{T, Array{T, 1}})
        if !isempty(vec)
            for j in eachindex(vec)
                Is[k] = start_row + 1
                Js[k] = j + start_col
                Vs[k] = vec[j]
                k += 1
            end
        end
        return k
    end

    function add_I_J_V(k, start_row, start_col, mat::SparseMatrixCSC{T, Int64})
        if !isempty(mat)
            for (i, j, v) in zip(findnz(mat)...)
                Is[k] = i + start_row
                Js[k] = j + start_col
                Vs[k] = v
                k += 1
            end
        end
        return k
    end

    rc1 = 0
    rc2 = n
    rc3 = n + p
    rc4 = n + p + q
    rc5 = n + p + q + 1
    rc6 = dim - 1

    offset = 1
    offset = add_I_J_V(offset, rc1, rc2, sparse(model.A'))
    offset = add_I_J_V(offset, rc1, rc3, sparse(model.G'))
    offset = add_I_J_V(offset, rc1, rc6, model.c)
    offset = add_I_J_V(offset, rc2, rc1, -sparse(model.A))
    offset = add_I_J_V(offset, rc2, rc6, model.b)
    offset = add_I_J_V(offset, rc4, rc4, [one(T)])
    system_solver.kap_idx = offset
    offset = add_I_J_V(offset, rc4, rc6, [one(T)])
    offset = add_I_J_V(offset, rc5, rc1, -sparse(model.G))
    offset = add_I_J_V(offset, rc5, rc5, -sparse(one(T) * I, q, q))
    offset = add_I_J_V(offset, rc5, rc6, model.h)
    offset = add_I_J_V(offset, rc6, rc1, -model.c')
    offset = add_I_J_V(offset, rc6, rc2, -model.b')
    offset = add_I_J_V(offset, rc6, rc3, -model.h')
    offset = add_I_J_V(offset, rc6, rc4, -[one(T)])

    H_indices = [Vector{Int}(undef, Cones.dimension(cone_k)) for cone_k in model.cones]
    for (k, cone_k) in enumerate(model.cones)
        cone_dim = Cones.dimension(cone_k)
        H_indices[k] = offset:(offset + cone_dim ^ 2 - 1)
        if Cones.use_dual(cone_k)
            offset = add_I_J_V(offset, rc3, rc3, sparse(ones(T, cone_dim, cone_dim)))
            offset = add_I_J_V(offset, rc3, rc5, sparse(one(T) * I, cone_dim, cone_dim))
        else
            offset = add_I_J_V(offset, rc3, rc5, sparse(ones(T, cone_dim, cone_dim)))
            offset = add_I_J_V(offset, rc3, rc3, sparse(one(T) * I, cone_dim, cone_dim))
        end
    end

    system_solver.lhs_copy = sparse(Is, Js, Vs)
    system_solver.lhs = similar(system_solver.lhs_copy)

    view_k(k::Int) = view(system_solver.Vs, H_indices[k])
    system_solver.lhs_H_Vs = [view_k(k) for k in eachindex(model.cones)]

    # function view_k(k::Int)
    #     rows = (n + p) .+ model.cone_idxs[k]
    #     cols = Cones.use_dual(model.cones[k]) ? rows : (q + 1) .+ rows
    #     return view(system_solver.lhs, rows, cols)
    # end
    # system_solver.lhs_H_k = [view_k(k) for k in eachindex(model.cones)]

    return system_solver
end

function get_combined_directions(system_solver::NaiveSparseSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    cones = model.cones
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    kap_row = system_solver.kap_row
    mu = solver.mu
    tau = solver.tau
    kap = solver.kap

    # update rhs matrix
    system_solver.x1 .= solver.x_residual
    system_solver.x2 .= zero(T)
    system_solver.y1 .= solver.y_residual
    system_solver.y2 .= zero(T)
    sqrtmu = sqrt(mu)
    for k in eachindex(cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cones[k])
        @. system_solver.z1_k[k] = -duals_k
        @. system_solver.z2_k[k] = -duals_k - grad_k * sqrtmu
    end
    system_solver.s1 .= solver.z_residual
    system_solver.s2 .= zero(T)
    rhs[kap_row, 1] = -kap
    rhs[kap_row, 2] = -kap + mu / tau
    rhs[end, 1] = kap + solver.primal_obj_t - solver.dual_obj_t
    rhs[end, 2] = zero(T)

    # solve system
    if system_solver.use_iterative
        lhs.blocks[end][1] = mu / tau / tau
        b_idx = 1
        for k in eachindex(cones)
            cone_k = cones[k]
            # TODO use hess prod instead

            lhs.blocks[b_idx + (Cones.use_dual(cone_k) ? 0 : 1)] = Cones.hess(cone_k)
            b_idx += 2
        end

        dim = size(lhs, 2)

        rhs1 = view(rhs, :, 1)
        IterativeSolvers.gmres!(system_solver.prevsol1, lhs, rhs1, restart = dim)
        copyto!(rhs1, system_solver.prevsol1)

        rhs2 = view(rhs, :, 2)
        IterativeSolvers.gmres!(system_solver.prevsol2, lhs, rhs2, restart = dim)
        copyto!(rhs2, system_solver.prevsol2)
    else
        # update lhs matrix
        # copyto!(lhs, system_solver.lhs_copy)
        # lhs[kap_row, end] = mu / tau / tau
        # for k in eachindex(cones)
        #     copyto!(system_solver.lhs_H_k[k], Cones.hess(cones[k]))
        # end

        for k in eachindex(cones)
            copyto!(system_solver.lhs_H_Vs[k], vec(Cones.hess(cones[k])))
        end
        system_solver.Vs[system_solver.kap_idx] = mu / tau / tau
        lhs = sparse(system_solver.Is, system_solver.Js, system_solver.Vs)
        # lhs[kap_row, end] = mu / tau / tau



        rhs .= lu(lhs) \ rhs
    end

    return (system_solver.x1, system_solver.x2, system_solver.y1, system_solver.y2, system_solver.z1, system_solver.z2, system_solver.s1, system_solver.s2, rhs[end, 1], rhs[end, 2], rhs[kap_row, 1], rhs[kap_row, 2])
end
