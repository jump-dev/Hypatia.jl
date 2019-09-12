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


using IterativeRefinement


mutable struct NaiveSystemSolver{T <: Real} <: SystemSolver{T}
    use_iterative::Bool
    use_sparse::Bool

    solver::Solver{T}

    lhs_copy
    lhs
    lhs_H_k
    rhs::Matrix{T}
    prevsol1::Vector{T}
    prevsol2::Vector{T}

    block_lhs

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



    # block matrix for efficient multiplication
    rc1 = 1:n
    rc2 = n .+ (1:p)
    rc3 = (n + p) .+ (1:q)
    rc4 = (n + p + q) .+ (1:1)
    rc5 = (n + p + q + 1) .+ (1:q)
    rc6 = dim:dim

    # TODO construct efficiently by preallocing the vectors
    cone_rows = UnitRange{Int}[]
    cone_cols = UnitRange{Int}[]
    cone_blocks = Any[]
    for k in eachindex(model.cones)
        rows = (n + p) .+ model.cone_idxs[k]
        push!(cone_rows, rows)
        push!(cone_rows, rows)
        push!(cone_cols, rows)
        push!(cone_cols, (q + 1) .+ rows)
        cone_k = model.cones[k]
        if Cones.use_dual(cone_k)
            push!(cone_blocks, cone_k)
            push!(cone_blocks, I)
        else
            push!(cone_blocks, I)
            push!(cone_blocks, cone_k)
        end
    end

    system_solver.block_lhs = BlockMatrix{T}(
        dim,
        dim,
        [cone_blocks..., model.A', model.G', reshape(model.c, :, 1), -model.A, reshape(model.b, :, 1), ones(T, 1, 1), -model.G, -I, reshape(model.h, :, 1), -model.c', -model.b', -model.h', -ones(T, 1, 1), ones(T, 1, 1)],
        [cone_rows..., rc1, rc1, rc1, rc2, rc2, rc4, rc5, rc5, rc5, rc6, rc6, rc6, rc6, rc4],
        [cone_cols..., rc2, rc3, rc6, rc1, rc6, rc4, rc1, rc5, rc6, rc1, rc2, rc3, rc4, rc6],
        )

    # x y z kap s tau
    if system_solver.use_iterative
        system_solver.prevsol1 = zeros(T, dim)
        system_solver.prevsol2 = zeros(T, dim)
    else
        if system_solver.use_sparse
            system_solver.lhs_copy = T[
                spzeros(T,n,n)  model.A'        model.G'              spzeros(T,n)  spzeros(T,n,q)         model.c;
                -model.A        spzeros(T,p,p)  spzeros(T,p,q)        spzeros(T,p)  spzeros(T,p,q)         model.b;
                spzeros(T,q,n)  spzeros(T,q,p)  sparse(one(T)*I,q,q)  spzeros(T,q)  sparse(one(T)*I,q,q)   spzeros(T,q);
                spzeros(T,1,n)  spzeros(T,1,p)  spzeros(T,1,q)        one(T)        spzeros(T,1,q)         one(T);
                -model.G        spzeros(T,q,p)  spzeros(T,q,q)        spzeros(T,q)  sparse(-one(T)*I,q,q)  model.h;
                -model.c'       -model.b'       -model.h'             -one(T)       spzeros(T,1,q)         zero(T);
                ]
            @assert issparse(system_solver.lhs_copy)
        else
            system_solver.lhs_copy = T[
                zeros(T,n,n)  model.A'      model.G'              zeros(T,n)  zeros(T,n,q)           model.c;
                -model.A      zeros(T,p,p)  zeros(T,p,q)          zeros(T,p)  zeros(T,p,q)           model.b;
                zeros(T,q,n)  zeros(T,q,p)  Matrix(one(T)*I,q,q)  zeros(T,q)  Matrix(one(T)*I,q,q)   zeros(T,q);
                zeros(T,1,n)  zeros(T,1,p)  zeros(T,1,q)          one(T)      zeros(T,1,q)           one(T);
                -model.G      zeros(T,q,p)  zeros(T,q,q)          zeros(T,q)  Matrix(-one(T)*I,q,q)  model.h;
                -model.c'     -model.b'     -model.h'             -one(T)     zeros(T,1,q)           zero(T);
                ]
        end

        system_solver.lhs = similar(system_solver.lhs_copy)
        function view_k(k::Int)
            rows = (n + p) .+ model.cone_idxs[k]
            cols = Cones.use_dual(model.cones[k]) ? rows : (q + 1) .+ rows
            return view(system_solver.lhs, rows, cols)
        end
        system_solver.lhs_H_k = [view_k(k) for k in eachindex(model.cones)]
    end

    return system_solver
end

function get_combined_directions(system_solver::NaiveSystemSolver{T}) where {T <: Real}
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


    block_lhs = system_solver.block_lhs

    mtt = mu / tau / tau
    block_lhs.blocks[end][1] = mtt # TODO only for residual



    # solve system
    if system_solver.use_iterative
        # TODO need preconditioner
        # TODO optimize for this case, including applying the blocks of the LHS
        # TODO pick which square non-symm method to use
        # TODO prealloc whatever is needed inside the solver
        # TODO possibly fix IterativeSolvers so that methods can take matrix RHS, however the two columns may take different number of iters needed to converge

        dim = size(lhs, 2)

        rhs1 = view(rhs, :, 1)
        IterativeSolvers.gmres!(system_solver.prevsol1, block_lhs, rhs1, restart = dim)
        copyto!(rhs1, system_solver.prevsol1)

        rhs2 = view(rhs, :, 2)
        IterativeSolvers.gmres!(system_solver.prevsol2, block_lhs, rhs2, restart = dim)
        copyto!(rhs2, system_solver.prevsol2)
    else
        # update lhs matrix
        copyto!(lhs, system_solver.lhs_copy)
        lhs[kap_row, end] = mtt
        for k in eachindex(cones)
            copyto!(system_solver.lhs_H_k[k], Cones.hess(cones[k]))
        end

        # lhs_copy = copy(lhs)
        # rhs_copy = copy(rhs)


        # if system_solver.use_sparse
        #     rhs .= lu(lhs) \ rhs
        # else
        #     ldiv!(lu!(lhs), rhs)
        # end


        if system_solver.use_sparse
            rhs .= lu(lhs) \ rhs
        else
            rhs_fix = copy(rhs)
            lhs_fix = copy(lhs)

            F = lu!(lhs)
            ldiv!(F, rhs)

            resi = BigFloat.(rhs_fix) - BigFloat.(lhs_fix) * BigFloat.(rhs)
            println("ldiv:  ", norm(resi, Inf), " ", norm(resi, 2))

            x, bnorm, bcomp = rfldiv(lhs_fix, rhs_fix)
            @show bnorm, bcomp
            rhs .= x
            resi = BigFloat.(rhs_fix) - BigFloat.(lhs_fix) * BigFloat.(rhs)
            println("rfld:  ", norm(resi, Inf), " ", norm(resi, 2))


            # if nres > 100 * eps(T)
            #     nresprev = nres
            #     nitref = 5
            #     for i in 1:nitref
            #         ldiv!(F, resi)
            #         # rhs_new = BigFloat.(rhs) + BigFloat.(resi)
            #         rhs_new = rhs + resi
            #         resi = rhs_fix - lhs_fix * rhs_new
            #         nres = norm(resi, Inf)
            #         if nres >= 0.8 * nresprev
            #             if nres < nresprev
            #                 rhs .= rhs_new
            #             end
            #             break
            #         end
            #         rhs .= rhs_new
            #         nresprev = nres
            #     end
            #
            #     resi = rhs_fix - lhs_fix * rhs
            #     println("iref:  ", norm(resi, Inf), " ", norm(resi, 2))
            # end

            # if nres > 100 * eps(T)
            #     dim = size(lhs, 2)
            #
            #     rhs_fix1 = view(rhs_fix, :, 1)
            #     rhs1 = view(rhs, :, 1)
            #     IterativeSolvers.gauss_seidel!(rhs1, block_lhs, rhs_fix1)
            #     # IterativeSolvers.gmres!(rhs1, block_lhs, rhs_fix1, restart = dim)
            #     # copyto!(rhs1, system_solver.prevsol1)
            #
            #     rhs_fix2 = view(rhs_fix, :, 2)
            #     rhs2 = view(rhs, :, 2)
            #     IterativeSolvers.gauss_seidel!(rhs2, block_lhs, rhs_fix2)
            #     # IterativeSolvers.gmres!(rhs2, block_lhs, rhs_fix2, restart = dim)
            #     # copyto!(rhs2, system_solver.prevsol2)
            #
            #     resi = rhs_fix - lhs_fix * rhs
            #     println("iterm: ", norm(resi, Inf), " ", norm(resi, 2))
            # end
        end

        println()


        # res = copy(rhs)
        # mul!(res, block_lhs, rhs)
        # @show norm(rhs_copy - res, Inf)
        #
        # iden = Matrix(Diagonal(1.0I, size(lhs, 1)))
        # testlhs = similar(iden)
        # mul!(testlhs, block_lhs, iden)
        # @show norm(testlhs - lhs_copy, Inf)
        # println()
    end

    return (system_solver.x1, system_solver.x2, system_solver.y1, system_solver.y2, system_solver.z1, system_solver.z2, system_solver.s1, system_solver.s2, rhs[end, 1], rhs[end, 2], rhs[kap_row, 1], rhs[kap_row, 2])
end


function setup_block_lhs(solver)
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)

    # block matrix for efficient multiplication
    rc1 = 1:n
    rc2 = n .+ (1:p)
    rc3 = (n + p) .+ (1:q)
    rc4 = (n + p + q) .+ (1:1)
    rc5 = (n + p + q + 1) .+ (1:q)
    dim = n + p + 2q + 2
    rc6 = dim:dim

    # TODO construct efficiently by preallocing the vectors
    cone_rows = UnitRange{Int}[]
    cone_cols = UnitRange{Int}[]
    cone_blocks = Any[]
    for k in eachindex(model.cones)
        rows = (n + p) .+ model.cone_idxs[k]
        push!(cone_rows, rows)
        push!(cone_rows, rows)
        push!(cone_cols, rows)
        push!(cone_cols, (q + 1) .+ rows)
        cone_k = model.cones[k]
        if Cones.use_dual(cone_k)
            push!(cone_blocks, cone_k)
            push!(cone_blocks, I)
        else
            push!(cone_blocks, I)
            push!(cone_blocks, cone_k)
        end
    end

    block_lhs = BlockMatrix{T}(
        dim,
        dim,
        [cone_blocks..., model.A', model.G', reshape(model.c, :, 1), -model.A, reshape(model.b, :, 1), ones(T, 1, 1), -model.G, -I, reshape(model.h, :, 1), -model.c', -model.b', -model.h', -ones(T, 1, 1), ones(T, 1, 1)],
        [cone_rows..., rc1, rc1, rc1, rc2, rc2, rc4, rc5, rc5, rc5, rc6, rc6, rc6, rc6, rc4],
        [cone_cols..., rc2, rc3, rc6, rc1, rc6, rc4, rc1, rc5, rc6, rc1, rc2, rc3, rc4, rc6],
        )

    return block_lhs
end


function calc_directions_residuals(x_pred, x_corr, y_pred, y_corr, z_pred, z_corr, s_pred, s_corr, tau_pred, tau_corr, kap_pred, kap_corr)



end



# TODO calc residuals of system
# TODO do steps iter ref in same precision until not improving
# TODO do gmres to polish solution until not improving

# TODO implement directions residual function more efficiently than blockmatrix
