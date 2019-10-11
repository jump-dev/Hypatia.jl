#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

naive linear system solver

6x6 nonsymmetric system in (x, y, z, tau, s, kap):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
-G*x + h*tau - s = zrhs
-c'*x - b'*y - h'*z - kap = taurhs
(pr bar) z_k + mu*H_k*s_k = srhs_k
(du bar) mu*H_k*z_k + s_k = srhs_k
mu/(taubar^2)*tau + kap = kaprhs
=#

abstract type NaiveNTSystemSolver{T <: Real} <: SystemSolver{T} end

#=
direct dense
=#

mutable struct NaiveDenseSystemSolver{T <: Real} <: NaiveNTSystemSolver{T}
    tau_row::Int
    lhs6::Matrix{T}
    lhs6_W_k::Vector
    lhs6_Wi_k::Vector
    fact_cache::DenseNonSymCache{T}
    function NaiveDenseSystemSolver{T}(; fact_cache::DenseNonSymCache{T} = DenseNonSymCache{T}()) where {T <: Real}
        system_solver = new{T}()
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::NaiveDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs
    system_solver.tau_row = n + p + q + 1

    system_solver.lhs6 = T[
        zeros(T, n, n)  model.A'        model.G'                  model.c      zeros(T, n, q)             zeros(T, n);
        -model.A        zeros(T, p, p)  zeros(T, p, q)            model.b      zeros(T, p, q)             zeros(T, p);
        -model.G        zeros(T, q, p)  zeros(T, q, q)            model.h      Matrix(-one(T) * I, q, q)  zeros(T, q);
        -model.c'       -model.b'       -model.h'                 zero(T)      zeros(T, 1, q)             -one(T);
        zeros(T, q, n)  zeros(T, q, p)  Matrix(one(T) * I, q, q)  zeros(T, q)  Matrix(one(T) * I, q, q)   zeros(T, q);
        zeros(T, 1, n)  zeros(T, 1, p)  zeros(T, 1, q)            one(T)       zeros(T, 1, q)             one(T);
        ]

    function view_W_k(cone_k, idxs_k)
        rows = system_solver.tau_row .+ idxs_k
        cols = (n + p) .+ idxs_k
        return view(system_solver.lhs6, rows, cols)
    end
    function view_Wi_k(cone_k, idxs_k)
        rows = system_solver.tau_row .+ idxs_k
        cols = rows
        return view(system_solver.lhs6, rows, cols)
    end
    system_solver.lhs6_W_k = [view_W_k(cone_k, idxs_k) for (cone_k, idxs_k) in zip(cones, cone_idxs)]
    system_solver.lhs6_Wi_k = [view_Wi_k(cone_k, idxs_k) for (cone_k, idxs_k) in zip(cones, cone_idxs)]

    load_matrix(system_solver.fact_cache, system_solver.lhs6)

    return system_solver
end

function update_fact(system_solver::NaiveDenseSystemSolver, solver::Solver)
    system_solver.lhs6[end, system_solver.tau_row] = solver.kappa
    system_solver.lhs6[end, end] = solver.tau

    for (k, cone_k) in enumerate(solver.model.cones)
        # W = Cones.scaling_matrix(cone_k)
        # Wi = Cones.scaling_matrix_inv(cone_k)
        # lambda = Cones.centered_point(cone_k)
        # prod = similar(W)
        # prod_i = similar(Wi)
        # lambda_dot_W = Cones.centered_point_prod(prod, W, cone_k)
        # lambda_dot_Wi = Cones.centered_point_prod(prod, Wi, cone_k)

        copyto!(system_solver.lhs6_W_k[k], Cones.lambda_W(cone_k))
        copyto!(system_solver.lhs6_Wi_k[k], Cones.lambda_Winv(cone_k))
    end

    update_fact(system_solver.fact_cache, system_solver.lhs6)

    return system_solver
end

function solve_system(system_solver::NaiveDenseSystemSolver, solver::Solver, sol6::Matrix, rhs6::Matrix)
    copyto!(sol6, rhs6)
    solve_system(system_solver.fact_cache, sol6)
    # TODO recover if fails - check issuccess
    return sol6
end
