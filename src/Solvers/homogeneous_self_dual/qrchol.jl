#=
Copyright 2018, Chris Coey and contributors

QR+Cholesky linear system solver
solves linear system in naive.jl via a procedure similar to that described by S10.3 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
=#

mutable struct QRCholCombinedHSDSystemSolver{T <: HypReal} <: CombinedHSDSystemSolver{T}
    use_sparse::Bool

    xi::Matrix{T}
    yi::Matrix{T}
    zi::Matrix{T}

    x1
    y1
    z1
    xi1
    xi2
    x2
    y2
    z2
    x3
    y3
    z3
    z_k
    z1_k
    z2_k
    z3_k
    zi_temp
    z1_temp
    z2_temp
    z3_temp
    z_temp_k
    z1_temp_k
    z2_temp_k
    z3_temp_k

    GQ1
    GQ2
    QpbxGHbz
    Q1pbxGHbz
    Q2pbxGHbz
    GQ1x
    HGQ1x
    Q2div
    HGQ2
    Q2GHGQ2
    Gxi
    HGxi

    HGQ1x_k
    GQ1x_k
    HGQ2_k
    GQ2_k
    HGxi_k
    Gxi_k

    function QRCholCombinedHSDSystemSolver{T}(model::Models.PreprocessedLinearModel{T}; use_sparse::Bool = false) where {T <: HypReal}
        (n, p, q) = (model.n, model.p, model.q)
        system_solver = new{T}()
        system_solver.use_sparse = use_sparse

        xi = Matrix{T}(undef, n, 3)
        yi = Matrix{T}(undef, p, 3)
        zi = Matrix{T}(undef, q, 3)
        system_solver.xi = xi
        system_solver.yi = yi
        system_solver.zi = zi

        system_solver.xi1 = view(xi, 1:p, :)
        system_solver.xi2 = view(xi, (p + 1):n, :)
        system_solver.x1 = view(xi, :, 1)
        system_solver.y1 = view(yi, :, 1)
        system_solver.z1 = view(zi, :, 1)
        system_solver.x2 = view(xi, :, 2)
        system_solver.y2 = view(yi, :, 2)
        system_solver.z2 = view(zi, :, 2)
        system_solver.x3 = view(xi, :, 3)
        system_solver.y3 = view(yi, :, 3)
        system_solver.z3 = view(zi, :, 3)
        system_solver.z_k = [view(zi, idxs, :) for idxs in model.cone_idxs]
        system_solver.z1_k = [view(zi, idxs, 1) for idxs in model.cone_idxs]
        system_solver.z2_k = [view(zi, idxs, 2) for idxs in model.cone_idxs]
        system_solver.z3_k = [view(zi, idxs, 3) for idxs in model.cone_idxs]
        zi_temp = similar(zi)
        system_solver.zi_temp = zi_temp
        system_solver.z1_temp = view(zi_temp, :, 1)
        system_solver.z2_temp = view(zi_temp, :, 2)
        system_solver.z3_temp = view(zi_temp, :, 3)
        system_solver.z_temp_k = [view(zi_temp, idxs, :) for idxs in model.cone_idxs]
        system_solver.z1_temp_k = [view(zi_temp, idxs, 1) for idxs in model.cone_idxs]
        system_solver.z2_temp_k = [view(zi_temp, idxs, 2) for idxs in model.cone_idxs]
        system_solver.z3_temp_k = [view(zi_temp, idxs, 3) for idxs in model.cone_idxs]

        nmp = n - p

        if !isa(model.G, Matrix{T}) && isa(model.Ap_Q, SuiteSparse.SPQR.QRSparseQ)
            # TODO very inefficient method used for sparse G * QRSparseQ : see https://github.com/JuliaLang/julia/issues/31124#issuecomment-501540818
            # TODO remove workaround and warning
            @warn("in QRChol, converting G to dense before multiplying by sparse Householder Q due to very inefficient dispatch")
            GQ = Matrix(model.G) * model.Ap_Q
        else
            GQ = model.G * model.Ap_Q
        end
        system_solver.GQ1 = GQ[:, 1:p]
        system_solver.GQ2 = GQ[:, (p + 1):end]
        if use_sparse
            if !issparse(GQ)
                error("to use sparse factorization for direction finding, cannot use dense A or G matrices (GQ is of type $(typeof(GQ)))")
            end
            system_solver.HGQ2 = spzeros(T, q, nmp)
            system_solver.Q2GHGQ2 = spzeros(T, nmp, nmp)
        else
            system_solver.HGQ2 = Matrix{T}(undef, q, nmp)
            system_solver.Q2GHGQ2 = Matrix{T}(undef, nmp, nmp)
        end
        system_solver.QpbxGHbz = Matrix{T}(undef, n, 3)
        system_solver.Q1pbxGHbz = view(system_solver.QpbxGHbz, 1:p, :)
        system_solver.Q2pbxGHbz = view(system_solver.QpbxGHbz, (p + 1):n, :)
        system_solver.GQ1x = Matrix{T}(undef, q, 3)
        system_solver.HGQ1x = similar(system_solver.GQ1x)
        system_solver.Q2div = Matrix{T}(undef, nmp, 3)
        system_solver.Gxi = similar(system_solver.GQ1x)
        system_solver.HGxi = similar(system_solver.Gxi)

        system_solver.HGQ1x_k = [view(system_solver.HGQ1x, idxs, :) for idxs in model.cone_idxs]
        system_solver.GQ1x_k = [view(system_solver.GQ1x, idxs, :) for idxs in model.cone_idxs]
        system_solver.HGQ2_k = [view(system_solver.HGQ2, idxs, :) for idxs in model.cone_idxs]
        system_solver.GQ2_k = [view(system_solver.GQ2, idxs, :) for idxs in model.cone_idxs]
        system_solver.HGxi_k = [view(system_solver.HGxi, idxs, :) for idxs in model.cone_idxs]
        system_solver.Gxi_k = [view(system_solver.Gxi, idxs, :) for idxs in model.cone_idxs]

        return system_solver
    end
end

function get_combined_directions(solver::HSDSolver{T}, system_solver::QRCholCombinedHSDSystemSolver{T}) where {T <: HypReal}
    model = solver.model
    cones = model.cones
    cone_idxs = model.cone_idxs
    mu = solver.mu

    xi = system_solver.xi
    yi = system_solver.yi
    zi = system_solver.zi
    xi1 = system_solver.xi1
    xi2 = system_solver.xi2
    x1 = system_solver.x1
    y1 = system_solver.y1
    z1 = system_solver.z1
    x2 = system_solver.x2
    y2 = system_solver.y2
    z2 = system_solver.z2
    x3 = system_solver.x3
    y3 = system_solver.y3
    z3 = system_solver.z3
    z_k = system_solver.z_k
    z1_k = system_solver.z1_k
    z2_k = system_solver.z2_k
    z3_k = system_solver.z3_k
    zi_temp = system_solver.zi_temp
    z1_temp = system_solver.z1_temp
    z2_temp = system_solver.z2_temp
    z3_temp = system_solver.z3_temp
    z_temp_k = system_solver.z_temp_k
    z1_temp_k = system_solver.z1_temp_k
    z2_temp_k = system_solver.z2_temp_k
    z3_temp_k = system_solver.z3_temp_k

    GQ1 = system_solver.GQ1
    GQ2 = system_solver.GQ2
    QpbxGHbz = system_solver.QpbxGHbz
    Q1pbxGHbz = system_solver.Q1pbxGHbz
    Q2pbxGHbz = system_solver.Q2pbxGHbz
    GQ1x = system_solver.GQ1x
    HGQ1x = system_solver.HGQ1x
    Q2div = system_solver.Q2div
    HGQ2 = system_solver.HGQ2
    Q2GHGQ2 = system_solver.Q2GHGQ2
    Gxi = system_solver.Gxi
    HGxi = system_solver.HGxi

    HGQ1x_k = system_solver.HGQ1x_k
    GQ1x_k = system_solver.GQ1x_k
    HGQ2_k = system_solver.HGQ2_k
    GQ2_k = system_solver.GQ2_k
    HGxi_k = system_solver.HGxi_k
    Gxi_k = system_solver.Gxi_k

    @timeit solver.timer "setup" begin

    @. x1 = -model.c
    @. x2 = solver.x_residual
    @. x3 = zero(T)
    @. y1 = model.b
    @. y2 = -solver.y_residual
    @. y3 = zero(T)

    @. z1_temp = model.h
    @. z2_temp = -solver.z_residual
    for k in eachindex(cones)
        cone_k = cones[k]
        duals_k = solver.point.dual_views[k]
        @timeit solver.timer "grad1" grad_k = Cones.grad(cone_k)
        if Cones.use_dual(cone_k)
            @. z1_temp_k[k] /= mu
            @. z2_temp_k[k] = (duals_k + z2_temp_k[k]) / mu
            @. z3_temp_k[k] = duals_k / mu + grad_k
            Cones.inv_hess_prod!(z_k[k], z_temp_k[k], cone_k)
        else
            @. z1_temp_k[k] *= mu
            @. z2_temp_k[k] *= mu
            @timeit solver.timer "hess1" Cones.hess_prod!(z1_k[k], z1_temp_k[k], cone_k)
            @timeit solver.timer "hess2" Cones.hess_prod!(z2_k[k], z2_temp_k[k], cone_k)
            @. z2_k[k] += duals_k
            @. z3_k[k] = duals_k + grad_k * mu
        end
    end

    end

    # TODO maybe replace with Diagonal(blocks) matrix product
    function block_hessian_product!(prod_k, arr_k)
        for k in eachindex(cones)
            cone_k = cones[k]
            if Cones.use_dual(cone_k)
                @timeit solver.timer "inv_hess_prod" begin
                Cones.inv_hess_prod!(prod_k[k], arr_k[k], cone_k)
                prod_k[k] ./= mu
                end
            else
                @timeit solver.timer "hess_prod" begin
                @timeit solver.timer "hess_prod1" Cones.hess_prod!(prod_k[k], arr_k[k], cone_k)
                @timeit solver.timer "hess_prod2" prod_k[k] .*= mu
                end
            end
        end
    end

    @timeit solver.timer "ldiv_yi" ldiv!(model.Ap_R', yi)

    @timeit solver.timer "QpbxGHbz" begin
    mul!(QpbxGHbz, model.G', zi)
    @. QpbxGHbz += xi
    lmul!(model.Ap_Q', QpbxGHbz)
    end

    if !iszero(size(Q2div, 1))
        @timeit solver.timer "Q2div" begin
        mul!(GQ1x, GQ1, yi)
        block_hessian_product!(HGQ1x_k, GQ1x_k)
        mul!(Q2div, GQ2', HGQ1x)
        @. Q2div = Q2pbxGHbz - Q2div
        end

        @timeit solver.timer "Hprod" block_hessian_product!(HGQ2_k, GQ2_k)
        @timeit solver.timer "GQ2prod" mul!(Q2GHGQ2, GQ2', HGQ2)

        if system_solver.use_sparse
            F = ldlt(Symmetric(Q2GHGQ2), check = false) # TODO not implemented for generic reals
            if !issuccess(F)
                println("sparse linear system matrix factorization failed")
                mul!(Q2GHGQ2, GQ2', HGQ2)
                F = ldlt(Symmetric(Q2GHGQ2), shift = T(1e-4), check = false)
                if !issuccess(F)
                    error("could not fix failure of positive definiteness (mu is $mu); terminating")
                end
            end
            Q2div .= F \ Q2div # TODO eliminate allocs (see https://github.com/JuliaLang/julia/issues/30084)
        else
            @timeit solver.timer "chol" F = hyp_chol!(Symmetric(Q2GHGQ2)) # TODO prealloc blasreal cholesky auxiliary vectors using posvx
            if !isposdef(F)
                println("dense linear system matrix factorization failed")
                mul!(Q2GHGQ2, GQ2', HGQ2)
                Q2GHGQ2 += T(1e-8) * I
                if T <: BlasReal
                    F = bunchkaufman!(Symmetric(Q2GHGQ2), true, check = false) # TODO prealloc with old sysvx code; not implemented for generic reals
                    # F = lu!(Symmetric(Q2GHGQ2), check = false) # TODO prealloc with old sysvx code; not implemented for generic reals
                    if !issuccess(F)
                        error("could not fix failure of positive definiteness (mu is $mu); terminating")
                    end
                else
                    F = hyp_chol!(Symmetric(Q2GHGQ2)) # TODO prealloc blasreal cholesky auxiliary vectors using posvx
                    if !isposdef(F)
                        error("could not fix failure of positive definiteness (mu is $mu); terminating")
                    end
                end
            end
            @timeit solver.timer "ldiv_Q2div" ldiv!(F, Q2div)
        end
    end

    @timeit solver.timer "xi" begin
    @. xi1 = yi
    @. xi2 = Q2div
    lmul!(model.Ap_Q, xi)
    end

    @timeit solver.timer "HGxi" begin
    mul!(Gxi, model.G, xi)
    block_hessian_product!(HGxi_k, Gxi_k)
    end

    @. zi = HGxi - zi

    if !iszero(length(yi))
        @timeit solver.timer "yi" begin
        mul!(yi, GQ1', HGxi)
        @. yi = Q1pbxGHbz - yi
        ldiv!(model.Ap_R, yi)
        end
    end

    # lift to HSDE space
    tau_denom = mu / solver.tau / solver.tau - dot(model.c, x1) - dot(model.b, y1) - dot(model.h, z1)

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

    (tau_pred, kap_pred) = lift!(x2, y2, z2, z2_temp, solver.kap + solver.primal_obj_t - solver.dual_obj_t, -solver.kap)
    @. z2_temp -= solver.z_residual
    (tau_corr, kap_corr) = lift!(x3, y3, z3, z3_temp, zero(T), -solver.kap + mu / solver.tau)

    return (x2, x3, y2, y3, z2, z3, z2_temp, z3_temp, tau_pred, tau_corr, kap_pred, kap_corr)
end
