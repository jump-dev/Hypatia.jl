#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

symmetric-indefinite linear system solver
solves linear system in naive.jl by first eliminating s and kap via the method in naiveelim.jl and then eliminating tau via a procedure similar to that described by S7.4 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

3x3 nonsymmetric system in (x, y, z):
A'*y + G'*z = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) mu*H_k*G_k*x - z_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]

multiply pr bar constraint by (mu*H_k)^-1 to get 3x3 symmetric indefinite system
A'*y + G'*z = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]

or to avoid inverse hessian products, let for pr bar w_k = (mu*H_k)\z_k (later recover z_k = mu*H_k*w_k) to get 3x3 symmetric indefinite system
A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
A*x = [-yrhs, b]
(pr bar) mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
(du bar) G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
=#

mutable struct SymIndefSparseExpandedSystemSolver{T <: Real} <: SymIndefSystemSolver{T}
    use_inv_hess::Bool
    tau_row::Int
    lhs3::SparseMatrixCSC # TODO type will depend on Int type
    other_lhs3::SparseMatrixCSC
    rhs3::Matrix{T}
    rhs3_expanded::Matrix{T}
    other_rhs3::Matrix{T}
    sol3::Matrix{T}
    sol3_expanded::Matrix{T}
    hess_idxs::Vector
    fact_cache::SparseSymCache{T}
    function SymIndefSparseExpandedSystemSolver{Float64}(;
        use_inv_hess::Bool = true,
        fact_cache::SparseSymCache{Float64} = SparseSymCache{Float64}(),
        )
        system_solver = new{Float64}()
        if !use_inv_hess
            @warn("SymIndefSparseExpandedSystemSolver is not implemented with `use_inv_hess` set to `false`, using `true` instead.")
        end
        system_solver.use_inv_hess = true
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function solve_system(system_solver::SymIndefSparseExpandedSystemSolver{T}, solver::Solver{T}, sol::Matrix{T}, rhs::Matrix{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = system_solver.tau_row

    sol3 = system_solver.sol3
    sol3_expanded = system_solver.sol3_expanded
    rhs3 = system_solver.rhs3_expanded

    @. @views rhs3[1:n, 1:2] = rhs[1:n, :]
    @. @views rhs3[n .+ (1:p), 1:2] = -rhs[n .+ (1:p), :]
    @. rhs3[1:n, 3] = -model.c
    @. rhs3[n .+ (1:p), 3] = model.b

    ns_added = 0
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        s_rows_k = tau_row .+ idxs_k
        zk12 = @view rhs[z_rows_k, :]
        sk12 = @view rhs[s_rows_k, :]
        hk = @view model.h[idxs_k]
        zk12_new = @view rhs3[z_rows_k .+ ns_added, 1:2]
        zk3_new = @view rhs3[z_rows_k .+ ns_added, 3]

        if Cones.use_dual(cone_k)
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            @. zk12_new = -zk12 - sk12
            @. zk3_new = hk
        elseif system_solver.use_inv_hess
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Cones.inv_hess_prod!(zk12_new, sk12, cone_k)
            @. zk12_new *= -1
            @. zk12_new -= zk12
            @. zk3_new = hk
        else
            # A'*y + sum_{pr bar} G_k'*mu*H_k*w_k + sum_{du bar} G_k'*z_k = [xrhs, -c]
            # mu*H_k*G_k*x - mu*H_k*w_k = [-mu*H_k*zrhs_k - srhs_k, mu*H_k*h_k]
            Cones.hess_prod!(zk12_new, zk12, cone_k)
            @. zk12_new *= -1
            @. zk12_new -= sk12
            Cones.hess_prod!(zk3_new, hk, cone_k)
        end
        if isa(cone_k, Cones.EpiNormEucl)
            ns_added += 1
        end
    end

    @timeit solver.timer "solve_system" solve_subsystem(system_solver, sol3_expanded, rhs3)

    @views copyto!(sol3[1:(n + p), :], sol3_expanded[1:(n + p), :])
    idx = n + p + 1
    ns_added = 0
    for (k, cone_k) in enumerate(model.cones)
        dim = Cones.dimension(cone_k)
        @views copyto!(sol3[idx:(idx + dim - 1), :], sol3_expanded[(idx + ns_added):(idx + ns_added + dim - 1), :])
        idx += dim
        if isa(cone_k, Cones.EpiNormEucl)
            ns_added += 1
        end
    end

    if !system_solver.use_inv_hess
        for (k, cone_k) in enumerate(model.cones)
            if !Cones.use_dual(cone_k)
                # recover z_k = mu*H_k*w_k
                z_rows_k = (n + p) .+ model.cone_idxs[k]
                z_copy_k = sol3[z_rows_k, :] # TODO do in-place
                @views Cones.hess_prod!(sol3[z_rows_k, :], z_copy_k, cone_k)
            end
        end
    end

    x3 = @view sol3[1:n, 3]
    y3 = @view sol3[n .+ (1:p), 3]
    z3 = @view sol3[(n + p) .+ (1:q), 3]
    x12 = @view sol3[1:n, 1:2]
    y12 = @view sol3[n .+ (1:p), 1:2]
    z12 = @view sol3[(n + p) .+ (1:q), 1:2]

    # lift to get tau
    # TODO maybe use higher precision here
    tau_denom = solver.mu / solver.tau / solver.tau - dot(model.c, x3) - dot(model.b, y3) - dot(model.h, z3)
    tau = @view sol[tau_row:tau_row, :]
    @. @views tau = rhs[tau_row:tau_row, :] + rhs[end:end, :]
    tau .+= model.c' * x12 + model.b' * y12 + model.h' * z12 # TODO in place
    @. tau /= tau_denom

    @. x12 += tau * x3
    @. y12 += tau * y3
    @. z12 += tau * z3

    @views sol[1:(tau_row - 1), :] = sol3[:, 1:2]

    # lift to get s and kap
    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol[(tau_row + 1):(end - 1), :]
    mul!(s, model.h, tau)
    mul!(s, model.G, sol[1:n, :], -one(T), true)
    @. @views s -= rhs[(n + p) .+ (1:q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs[end:end, :]

    return sol
end

function load(system_solver::SymIndefSparseExpandedSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.fact_cache.analyzed = false
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    system_solver.tau_row = n + p + q + 1
    cones = model.cones
    cone_idxs = model.cone_idxs

    ns = sum(isa(cone_k, Cones.EpiNormEucl) for cone_k in cones)
    qns = q + ns

    system_solver.sol3 = zeros(n + p + q, 3)
    system_solver.sol3_expanded = zeros(n + p + q + ns, 3)
    system_solver.rhs3 = similar(system_solver.sol3)
    system_solver.rhs3_expanded = zeros(n + p + q + ns, 3)

    # form sparse LHS without Hessians and inverse Hessians in z/z block
    lhs3 = T[
        spzeros(T, n, n)   spzeros(T, n, p)    spzeros(T, n, qns);
        model.A            spzeros(T, p, p)    spzeros(T, p, qns);
        spzeros(T, qns, n) spzeros(T, qns, p)  sparse(-one(T) * I, qns, qns);
        ]
    idx = 1
    ns_added = 0
    for (k, cone_k) in enumerate(model.cones)
        dim = Cones.dimension(cone_k)
        @. @views lhs3[(n + p + idx + ns_added):(n + p + idx + dim - 1 + ns_added), 1:n] = model.G[idx:(idx + dim - 1), :]
        idx += dim
        if isa(cone_k, Cones.EpiNormEucl)
            ns_added += 1
        end
    end

    @assert issparse(lhs3)
    dropzeros!(lhs3)
    (Is, Js, Vs) = findnz(lhs3)
    pert = T(system_solver.fact_cache.diag_pert) # TODO not happy with where this is stored
    append!(Is, 1:(n + p))
    append!(Js, 1:(n + p))
    append!(Vs, fill(pert, n))
    append!(Vs, fill(-pert, p))

    dim = size(lhs3, 1)
    # integer type supported by the sparse system solver library to be used
    Ti = int_type(system_solver.fact_cache)
    # prefer conversions of integer types to happen here than inside external wrappers
    Is = convert(Vector{Ti}, Is)
    Js = convert(Vector{Ti}, Js)
    lhs3 = system_solver.lhs3 = sparse(Is, Js, Vs, dim, dim)

    return system_solver
end

function update_fact(system_solver::SymIndefSparseExpandedSystemSolver, solver::Solver)
    idx = solver.model.n + solver.model.p + 1
    for (k, cone_k) in enumerate(solver.model.cones)
        dim = Cones.dimension(cone_k)
        if isa(cone_k, Cones.EpiNormEucl)
            if Cones.use_dual(cone_k)
                vec = cone_k.grad
                scal = inv(cone_k.dist)
            else
                vec = cone_k.point
                scal = cone_k.dist
            end
            # Cones.update_feas(cone_k)
            system_solver.lhs3[idx, idx] = scal
            for j in (idx + 1):(idx + dim - 1)
                system_solver.lhs3[j, j] = -scal
            end
            system_solver.lhs3[(idx + dim), (idx + dim)] = 1
            system_solver.lhs3[(idx + dim), idx:(idx + dim - 1)] .= -vec
            idx += (dim + 1)
        else
            H = (Cones.use_dual(cone_k) ? Cones.hess(cone_k) : Cones.inv_hess(cone_k))
            system_solver.lhs3[idx:(idx + dim - 1), idx:(idx + dim - 1)] .= -H
            idx += dim
        end
    end

    @timeit solver.timer "update_fact" update_fact(system_solver.fact_cache, system_solver.lhs3)

    return system_solver
end

function solve_subsystem(system_solver::SymIndefSparseExpandedSystemSolver, sol3_expanded::Matrix, rhs3::Matrix)
    solve_system(system_solver.fact_cache, sol3_expanded, system_solver.lhs3, rhs3)
    # sol3_expanded .= Symmetric(system_solver.lhs3, :L) \ rhs3

    return sol3_expanded
end
