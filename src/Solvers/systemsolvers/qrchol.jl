#=
Copyright 2018, Chris Coey and contributors

QR+Cholesky linear system solver
requires precomputed QR factorization of A'

solves linear system in naive.jl by first eliminating s, kap, and tau via the method in the symindef solver, then reducing the 3x3 symmetric indefinite system to a series of low-dimensional operations via a procedure similar to that described by S10.3 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf (the dominating subroutine is a positive definite linear solve with RHS of dimension n-p x 3)
=#

abstract type QRCholSystemSolver{T <: Real} <: SystemSolver{T} end

function solve_system(system_solver::QRCholSystemSolver{T}, solver::Solver{T}, sol::Vector{T}, rhs::Vector{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    @timeit solver.timer "setup_rhs3" begin
        rhs3 = system_solver.rhs3
        dim3 = length(rhs3)
        x_rows = 1:n
        y_rows = n .+ (1:p)
        z_rows = (n + p) .+ (1:q)

        @. @views rhs3[x_rows] = rhs[x_rows]
        @. @views rhs3[y_rows] = -rhs[y_rows]

        for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
            z_rows_k = (n + p) .+ idxs_k
            z_k = @view rhs[z_rows_k]
            z3_k = @view rhs3[z_rows_k]
            s_k = @view rhs[(dim3 + 1) .+ idxs_k]

            if Cones.use_dual_barrier(cone_k)
                z_temp_k = @view sol[z_rows_k]
                @. z_temp_k = -z_k - s_k
                Cones.inv_hess_prod!(z3_k, z_temp_k, cone_k)
                z3_k ./= solver.mu
            else
                Cones.hess_prod!(z3_k, z_k, cone_k)
                axpby!(-1, s_k, -solver.mu, z3_k)
            end
        end
    end

    @timeit solver.timer "solve_subsystem" sol3 = solve_subsystem(system_solver, solver, rhs3) # NOTE modifies and returns rhs3

    @timeit solver.timer "lift_sol3" begin
        # TODO refactor all below
        # TODO maybe use higher precision here
        const_sol = system_solver.const_sol

        # lift to get tau
        @views tau_num = rhs[dim3 + 1] + rhs[end] + dot(model.c, sol3[x_rows]) + dot(model.b, sol3[y_rows]) + dot(model.h, sol3[z_rows])
        @views tau_denom = solver.mu / solver.tau / solver.tau - dot(model.c, const_sol[x_rows]) - dot(model.b, const_sol[y_rows]) - dot(model.h, const_sol[z_rows])

        sol_tau = tau_num / tau_denom
        @. sol[1:dim3] = sol3 + sol_tau * const_sol
        sol[dim3 + 1] = sol_tau

        # lift to get s and kap
        # TODO refactor below for use with symindef and qrchol methods
        s = @view sol[(dim3 + 2):(end - 1)]
        @. @views s = model.h * sol_tau - rhs[z_rows]
        @views mul!(s, model.G, sol[x_rows], -1, true)

        # kap = -mu/(taubar^2)*tau + kaprhs
        sol[end] = -solver.mu / solver.tau * sol_tau / solver.tau + rhs[end]
    end

    return sol
end

function solve_subsystem(system_solver::QRCholSystemSolver{T}, solver::Solver{T}, rhs3::Vector{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    @timeit solver.timer "setup_rhs_sub" begin
        x = @view rhs3[1:n]
        x_sub1 = @view rhs3[1:p]
        x_sub2 = @view rhs3[(p + 1):n]
        y = @view rhs3[n .+ (1:p)]
        z = @view rhs3[(n + p) .+ (1:q)]

        ldiv!(solver.Ap_R', y)

        copyto!(system_solver.QpbxGHbz, x) # TODO can be avoided
        mul!(system_solver.QpbxGHbz, model.G', z, true, true)
        lmul!(solver.Ap_Q', system_solver.QpbxGHbz)

        copyto!(x_sub1, y)
    end

    if !isempty(system_solver.Q2div)
        @timeit solver.timer "solve_sub" begin
            mul!(system_solver.GQ1x, system_solver.GQ1, y)
            block_hess_prod.(model.cones, system_solver.HGQ1x_k, system_solver.GQ1x_k, solver.mu)
            mul!(system_solver.Q2div, system_solver.GQ2', system_solver.HGQ1x, -1, true)
            copyto!(x_sub2, system_solver.Q2div)
            inv_prod(system_solver.fact_cache, x_sub2)
        end
    end

    @timeit solver.timer "lift_sol_sub" begin
        lmul!(solver.Ap_Q, x)

        mul!(system_solver.Gx, model.G, x)
        block_hess_prod.(model.cones, system_solver.HGx_k, system_solver.Gx_k, solver.mu)

        @. z = system_solver.HGx - z

        if !isempty(y)
            copyto!(y, system_solver.Q1pbxGHbz)
            mul!(y, system_solver.GQ1', system_solver.HGx, -1, true)
            ldiv!(solver.Ap_R, y)
        end
    end

    return rhs3
end

function block_hess_prod(cone_k::Cones.Cone{T}, prod_k::AbstractVecOrMat{T}, arr_k::AbstractVecOrMat{T}, mu::T) where {T <: Real}
    if Cones.use_dual_barrier(cone_k)
        Cones.inv_hess_prod!(prod_k, arr_k, cone_k)
        @. prod_k /= mu
    else
        Cones.hess_prod!(prod_k, arr_k, cone_k)
        @. prod_k *= mu
    end
    return
end

#=
direct dense
=#

mutable struct QRCholDenseSystemSolver{T <: Real} <: QRCholSystemSolver{T}
    rhs3::Vector{T}
    const_sol::Vector{T}
    lhs1::Symmetric{T, Matrix{T}}
    GQ1
    GQ2
    QpbxGHbz
    Q1pbxGHbz
    Q2div
    GQ1x
    HGQ1x
    HGQ2
    Gx
    HGx
    HGQ1x_k
    GQ1x_k
    HGQ2_k
    GQ2_k
    HGx_k
    Gx_k
    fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} # can use BunchKaufman or Cholesky
    function QRCholDenseSystemSolver{T}(;
        fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} = DensePosDefCache{T}(),
        # fact_cache::Union{DensePosDefCache{T}, DenseSymCache{T}} = DenseSymCache{T}(),
        ) where {T <: Real}
        system_solver = new{T}()
        system_solver.fact_cache = fact_cache # TODO start with cholesky and then switch to BK if numerical issues
        return system_solver
    end
end

function load(system_solver::QRCholDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cone_idxs = model.cone_idxs

    system_solver.rhs3 = zeros(T, n + p + q)

    # TODO optimize for case of empty A
    # TODO very inefficient method used for sparse G * QRSparseQ : see https://github.com/JuliaLang/julia/issues/31124#issuecomment-501540818
    if !isa(model.G, Matrix{T})
        @warn("in QRChol, converting G to dense before multiplying by sparse Householder Q due to very inefficient dispatch")
    end
    G = Matrix(model.G)
    GQ = rmul!(G, solver.Ap_Q)

    system_solver.GQ1 = GQ[:, 1:p]
    system_solver.GQ2 = GQ[:, (p + 1):end]
    nmp = n - p
    system_solver.HGQ2 = Matrix{T}(undef, q, nmp)
    system_solver.lhs1 = Symmetric(Matrix{T}(undef, nmp, nmp), :U)
    system_solver.QpbxGHbz = Vector{T}(undef, n)
    system_solver.Q1pbxGHbz = view(system_solver.QpbxGHbz, 1:p)
    system_solver.Q2div = view(system_solver.QpbxGHbz, (p + 1):n)
    system_solver.GQ1x = Vector{T}(undef, q)
    system_solver.HGQ1x = similar(system_solver.GQ1x)
    system_solver.Gx = similar(system_solver.GQ1x)
    system_solver.HGx = similar(system_solver.Gx)
    system_solver.HGQ1x_k = [view(system_solver.HGQ1x, idxs, :) for idxs in cone_idxs]
    system_solver.GQ1x_k = [view(system_solver.GQ1x, idxs, :) for idxs in cone_idxs]
    system_solver.HGQ2_k = [view(system_solver.HGQ2, idxs, :) for idxs in cone_idxs]
    system_solver.GQ2_k = [view(system_solver.GQ2, idxs, :) for idxs in cone_idxs]
    system_solver.HGx_k = [view(system_solver.HGx, idxs, :) for idxs in cone_idxs]
    system_solver.Gx_k = [view(system_solver.Gx, idxs, :) for idxs in cone_idxs]

    load_matrix(system_solver.fact_cache, system_solver.lhs1)

    system_solver.const_sol = similar(system_solver.rhs3)

    return system_solver
end

# NOTE move to dense.jl if useful elsewhere
outer_prod(A::AbstractMatrix{T}, B::AbstractMatrix{T}, alpha::Real, beta::Real) where {T <: LinearAlgebra.BlasReal} = BLAS.syrk!('U', 'T', alpha, A, beta, B)
outer_prod(A::AbstractMatrix{T}, B::AbstractMatrix{T}, alpha::Real, beta::Real) where {T <: Real} = mul!(B, A', A, alpha, beta)

function update_lhs(system_solver::QRCholDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    timer = solver.timer
    lhs = system_solver.lhs1.data

    if !isempty(system_solver.Q2div)
        @timeit timer "update_inner_lhs" begin
            inv_hess_cones = Int[]
            inv_hess_sqrt_cones = Int[]
            hess_cones = Int[]
            hess_sqrt_cones = Int[]

            # update hessian factorizations and partition of cones
            for (k, cone_k) in enumerate(model.cones)
                if hasfield(typeof(cone_k), :hess_fact_cache) # TODO use dispatch or a function
                    @timeit timer "update_hess_fact" Cones.update_hess_fact(cone_k)
                    if cone_k.hess_fact_cache isa DenseSymCache{T}
                        cones_list = Cones.use_dual_barrier(cone_k) ? inv_hess_cones : hess_cones
                        push!(cones_list, k)
                        continue
                    end
                end
                cones_list = Cones.use_dual_barrier(cone_k) ? inv_hess_sqrt_cones : hess_sqrt_cones
                push!(cones_list, k)
            end

            # do inv_hess and inv_hess_sqrt cones
            if isempty(inv_hess_sqrt_cones)
                lhs .= 0
            else
                idx = 1
                for k in inv_hess_sqrt_cones
                    arr_k = system_solver.GQ2_k[k]
                    q_k = size(arr_k, 1)
                    @views prod_k = system_solver.HGQ2[idx:(idx + q_k - 1), :]
                    @timeit timer "inv_hess_sqrt" Cones.inv_hess_sqrt_prod!(prod_k, arr_k, model.cones[k])
                    idx += q_k
                end
                @views HGQ2_sub = system_solver.HGQ2[1:(idx - 1), :]
                @timeit timer "syrk_inv_hess_sqrt" outer_prod(HGQ2_sub, lhs, true, false)
            end

            for k in inv_hess_cones
                arr_k = system_solver.GQ2_k[k]
                prod_k = system_solver.HGQ2_k[k]
                @timeit timer "inv_hess" Cones.inv_hess_prod!(prod_k, arr_k, model.cones[k])
                @timeit timer "mul" mul!(lhs, arr_k', prod_k, true, true)
            end

            if !(isempty(inv_hess_cones) && isempty(inv_hess_sqrt_cones))
                # divide by mu for inv_hess and inv_hess_sqrt cones
                lhs ./= solver.mu
            end

            # do hess and hess_sqrt cones
            if !isempty(hess_sqrt_cones)
                idx = 1
                for k in hess_sqrt_cones
                    arr_k = system_solver.GQ2_k[k]
                    q_k = size(arr_k, 1)
                    @views prod_k = system_solver.HGQ2[idx:(idx + q_k - 1), :]
                    @timeit timer "hess_sqrt" Cones.hess_sqrt_prod!(prod_k, arr_k, model.cones[k])
                    idx += q_k
                end
                @views HGQ2_sub = system_solver.HGQ2[1:(idx - 1), :]
                @timeit timer "syrk_hess_sqrt" outer_prod(HGQ2_sub, lhs, solver.mu, true)
            end

            for k in hess_cones
                arr_k = system_solver.GQ2_k[k]
                prod_k = system_solver.HGQ2_k[k]
                @timeit timer "hess" Cones.hess_prod!(prod_k, arr_k, model.cones[k])
                @timeit timer "mul" mul!(lhs, arr_k', prod_k, solver.mu, true)
            end
        end

        # TODO refactor below
        @timeit timer "update_fact" success = update_fact(system_solver.fact_cache, system_solver.lhs1)
        if !success
            @timeit timer "recover" begin
                @warn("QRChol factorization failed")
                if T <: LinearAlgebra.BlasReal && system_solver.fact_cache isa DensePosDefCache{T}
                    @warn("switching QRChol solver from Cholesky to Bunch Kaufman")
                    system_solver.fact_cache = DenseSymCache{T}()
                    load_matrix(system_solver.fact_cache, system_solver.lhs1)
                else
                    system_solver.lhs1 += sqrt(eps(T)) * I # attempt recovery # TODO make more efficient
                end
                @timeit timer "update_fact" success = update_fact(system_solver.fact_cache, system_solver.lhs1)
                success || @warn("QRChol Bunch-Kaufman factorization failed after recovery")
            end
        end
    end

    # update solution for fixed c,b,h part
    @timeit timer "update_fixed_rhs" begin
        (n, p) = (model.n, model.p)
        const_sol = system_solver.const_sol
        @views const_sol[1:n] = -model.c
        @views const_sol[n .+ (1:p)] = model.b
        for (cone_k, idxs_k) in zip(model.cones, model.cone_idxs)
            @views block_hess_prod(cone_k, const_sol[(n + p) .+ idxs_k], model.h[idxs_k], solver.mu)
        end
    end
    @timeit timer "solve_subsystem" solve_subsystem(system_solver, solver, const_sol)

    return system_solver
end
