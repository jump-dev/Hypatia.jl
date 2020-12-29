#=
naive+elimination linear system solver
solves linear system in naive.jl via the following procedure

eliminate s
-G*x + h*tau - s = zrhs
so if using primal barrier
z_k + mu*H_k*s_k = srhs_k --> s_k = (mu*H_k)\(srhs_k - z_k)
-->
-G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k (if use_inv_hess = true)
-->
-mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k (if use_inv_hess = false)
or if using dual barrier
mu*H_k*z_k + s_k = srhs_k --> s_k = srhs_k - mu*H_k*z_k
-->
-G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k

eliminate kap
-c'x - b'y - h'z - kap = taurhs
so
mu/(taubar^2)*tau + kap = kaprhs --> kap = kaprhs - mu/(taubar^2)*tau
-->
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs

4x4 nonsymmetric system in (x, y, z, tau):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
(pr bar) -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
(du bar) -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs
=#

abstract type NaiveElimSystemSolver{T <: Real} <: SystemSolver{T} end

function solve_subsystem4(
    system_solver::NaiveElimSystemSolver{T},
    solver::Solver{T},
    sol::Point{T},
    rhs::Point{T},
    tau_scal::T,
    ) where {T <: Real}
    rhs_sub = system_solver.rhs_sub
    dim4 = length(rhs_sub.vec)
    @inbounds @views rhs_sub.vec .= rhs.vec[1:dim4]

    @inbounds for (k, cone_k) in enumerate(solver.model.cones)
        rhs_z_k = rhs.z_views[k]
        rhs_s_k = rhs.s_views[k]
        rhs_sub_z_k = rhs_sub.z_views[k]
        if Cones.use_dual_barrier(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            rhs_sub_z_k .+= rhs_s_k
        elseif system_solver.use_inv_hess
            # -G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k
            Cones.inv_hess_prod!(rhs_sub_z_k, rhs_s_k, cone_k)
            rhs_sub_z_k .+= rhs_z_k
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            Cones.hess_prod!(rhs_sub_z_k, rhs_z_k, cone_k)
            rhs_sub_z_k .+= rhs_s_k
        end
    end
    # -c'x - b'y - h'z + kapbar/taubar*tau = taurhs + kaprhs
    rhs_sub.vec[end] += rhs.kap[]

    sol_sub = system_solver.sol_sub
    solve_inner_system(system_solver, sol_sub, rhs_sub)
    @inbounds @views sol.vec[1:dim4] .= sol_sub.vec

    return sol
end

function setup_point_sub(system_solver::NaiveElimSystemSolver{T}, model::Models.Model{T}) where {T <: Real}
    system_solver.sol_sub = Point{T}()
    system_solver.rhs_sub = Point{T}()
    z_start = model.n + model.p
    dim_sub = size(system_solver.lhs_sub, 1)
    for point_sub in (system_solver.sol_sub, system_solver.rhs_sub)
        point_sub.vec = zeros(T, dim_sub)
        point_sub.z_views = [view(point_sub.vec, z_start .+ idxs) for idxs in model.cone_idxs]
    end
    return nothing
end

#=
direct sparse
=#

mutable struct NaiveElimSparseSystemSolver{T <: Real} <: NaiveElimSystemSolver{T}
    use_inv_hess::Bool
    lhs_sub::SparseMatrixCSC # TODO type params will depend on factor cache Int type
    fact_cache::SparseNonSymCache{T}
    rhs_sub::Point{T}
    sol_sub::Point{T}
    hess_idxs::Vector
    function NaiveElimSparseSystemSolver{T}(;
        use_inv_hess::Bool = true,
        fact_cache::SparseNonSymCache{T} = SparseNonSymCache{T}(),
        ) where {T <: Real}
        system_solver = new{T}()
        if !use_inv_hess
            @warn("SymIndefSparseSystemSolver is not implemented with `use_inv_hess` set to `false`, using `true` instead.")
        end
        system_solver.use_inv_hess = use_inv_hess
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::NaiveElimSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.fact_cache.analyzed = false
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs

    # form sparse LHS without Hessians and inverse Hessians in z/z block
    lhs_sub = T[
        spzeros(T, n, n)  model.A'          model.G'                  model.c;
        -model.A          spzeros(T, p, p)  spzeros(T, p, q)          model.b;
        -model.G          spzeros(T, q, p)  sparse(one(T) * I, q, q)  model.h;
        -model.c'         -model.b'         -model.h'                 one(T);
        ]
    dropzeros!(lhs_sub)
    (Is, Js, Vs) = findnz(lhs_sub)

    # add I, J, V for Hessians and inverse Hessians
    if isempty(cones)
        hess_nz_total = 0
    else
        hess_nz_total = sum(Cones.use_dual_barrier(cone_k) ? Cones.hess_nz_count(cone_k) : Cones.inv_hess_nz_count(cone_k) for cone_k in cones)
    end
    H_Is = Vector{Int}(undef, hess_nz_total)
    H_Js = Vector{Int}(undef, hess_nz_total)
    offset = 1
    for (cone_k, idxs_k) in zip(cones, cone_idxs)
        z_start_k = n + p + first(idxs_k) - 1
        for j in 1:Cones.dimension(cone_k)
            nz_rows_kj = z_start_k .+ (Cones.use_dual_barrier(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j) : Cones.inv_hess_nz_idxs_col(cone_k, j))
            len_kj = length(nz_rows_kj)
            IJV_idxs = offset:(offset + len_kj - 1)
            offset += len_kj
            @. @views H_Is[IJV_idxs] = nz_rows_kj
            @. @views H_Js[IJV_idxs] = z_start_k + j
        end
    end
    append!(Is, H_Is)
    append!(Js, H_Js)
    append!(Vs, ones(T, hess_nz_total))

    # prefer conversions of integer types to happen here than inside external wrappers
    # integer type supported by the sparse system solver library to be used
    Ti = int_type(system_solver.fact_cache)
    Is = convert(Vector{Ti}, Is)
    Js = convert(Vector{Ti}, Js)
    dim = size(lhs_sub, 1)
    lhs_sub = system_solver.lhs_sub = sparse(Is, Js, Vs, dim, dim)

    # cache indices of nonzeros of Hessians and inverse Hessians in sparse LHS nonzeros vector
    system_solver.hess_idxs = [Vector{Union{UnitRange, Vector{Int}}}(undef, Cones.dimension(cone_k)) for cone_k in cones]
    for (k, cone_k) in enumerate(cones)
        z_start_k = n + p + first(cone_idxs[k]) - 1
        for j in 1:Cones.dimension(cone_k)
            col = z_start_k + j
            # get nonzero rows in the current column of the LHS
            col_idx_start = lhs_sub.colptr[col]
            nz_rows = lhs_sub.rowval[col_idx_start:(lhs_sub.colptr[col + 1] - 1)]
            # get nonzero rows in column j of the Hessian or inverse Hessian
            nz_hess_indices = (Cones.use_dual_barrier(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j) : Cones.inv_hess_nz_idxs_col(cone_k, j))
            # get index corresponding to first nonzero Hessian element of the current column of the LHS
            first_H = findfirst(isequal(z_start_k + first(nz_hess_indices)), nz_rows)
            # indices of nonzero values for cone k column j
            system_solver.hess_idxs[k][j] = (col_idx_start + first_H - 2) .+ (1:length(nz_hess_indices))
        end
    end

    setup_point_sub(system_solver, model)

    return system_solver
end

function update_lhs(system_solver::NaiveElimSparseSystemSolver, solver::Solver)
    for (k, cone_k) in enumerate(solver.model.cones)
        H_k = (Cones.use_dual_barrier(cone_k) ? Cones.hess(cone_k) : Cones.inv_hess(cone_k))
        for j in 1:Cones.dimension(cone_k)
            nz_rows = (Cones.use_dual_barrier(cone_k) ? Cones.hess_nz_idxs_col(cone_k, j) : Cones.inv_hess_nz_idxs_col(cone_k, j))
            @views copyto!(system_solver.lhs_sub.nzval[system_solver.hess_idxs[k][j]], H_k[nz_rows, j])
        end
    end
    tau = solver.point.tau[]
    system_solver.lhs_sub.nzval[end] = solver.mu / tau / tau # NOTE: mismatch when using NT for kaptau

    update_fact(system_solver.fact_cache, system_solver.lhs_sub)

    return system_solver
end

#=
direct dense
=#

mutable struct NaiveElimDenseSystemSolver{T <: Real} <: NaiveElimSystemSolver{T}
    use_inv_hess::Bool
    lhs_sub::Matrix{T}
    fact_cache::DenseNonSymCache{T}
    rhs_sub::Point{T}
    sol_sub::Point{T}
    function NaiveElimDenseSystemSolver{T}(;
        use_inv_hess::Bool = true,
        fact_cache::DenseNonSymCache{T} = DenseNonSymCache{T}(),
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_inv_hess = use_inv_hess
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::NaiveElimDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    system_solver.lhs_sub = T[
        zeros(T, n, n)  model.A'        model.G'                  model.c;
        -model.A        zeros(T, p, p)  zeros(T, p, q)            model.b;
        -model.G        zeros(T, q, p)  Matrix(one(T) * I, q, q)  model.h;
        -model.c'       -model.b'       -model.h'                 one(T);
        ]

    load_matrix(system_solver.fact_cache, system_solver.lhs_sub)

    setup_point_sub(system_solver, model)

    return system_solver
end

function update_lhs(system_solver::NaiveElimDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p) = (model.n, model.p)
    lhs_sub = system_solver.lhs_sub

    @inbounds for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
        z_rows_k = (n + p) .+ idxs_k
        if Cones.use_dual_barrier(cone_k)
            # -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
            @views copyto!(lhs_sub[z_rows_k, z_rows_k], Cones.hess(cone_k))
        elseif system_solver.use_inv_hess
            # -G_k*x + (mu*H_k)\z_k + h_k*tau = zrhs_k + (mu*H_k)\srhs_k
            Hi_k = @views copyto!(lhs_sub[z_rows_k, z_rows_k], Cones.inv_hess(cone_k))
        else
            # -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
            @views Cones.hess_prod!(lhs_sub[z_rows_k, 1:n], model.G[idxs_k, :], cone_k)
            @. @views lhs_sub[z_rows_k, 1:n] *= -1
            @views Cones.hess_prod!(lhs_sub[z_rows_k, end], model.h[idxs_k], cone_k)
        end
    end
    tau = solver.point.tau[]
    lhs_sub[end, end] = solver.mu / tau / tau # NOTE: mismatch when using NT for kaptau

    update_fact(system_solver.fact_cache, system_solver.lhs_sub)

    return system_solver
end
