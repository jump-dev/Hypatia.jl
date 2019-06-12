#=
Copyright 2018, Chris Coey and contributors

QR + Cholesky linear system solver
=#

mutable struct QRCholCombinedHSDSystemSolver{T <: HypReal} <: CombinedHSDSystemSolver{T}
    use_sparse::Bool

    xi::Matrix{T}
    yi::Matrix{T}
    zi::Matrix{T}

    x1
    y1
    z1
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

    bxGHbz
    Q1x
    GQ1x
    HGQ1x
    GHGQ1x
    Q2div

    GQ1
    GQ2

    HGQ2
    Q2GHGQ2
    Q2x
    Gxi
    HGxi
    GHGxi

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

        system_solver.bxGHbz = Matrix{T}(undef, n, 3)
        system_solver.Q1x = similar(system_solver.bxGHbz)
        system_solver.GQ1x = Matrix{T}(undef, q, 3)
        system_solver.HGQ1x = similar(system_solver.GQ1x)
        system_solver.GHGQ1x = Matrix{T}(undef, n, 3)
        system_solver.Q2div = Matrix{T}(undef, nmp, 3)

        GQ = model.G * model.Ap_Q # TODO check efficiency
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

        system_solver.Q2x = similar(system_solver.Q1x)
        system_solver.Gxi = similar(system_solver.GQ1x)
        system_solver.HGxi = similar(system_solver.Gxi)
        system_solver.GHGxi = similar(system_solver.GHGQ1x)

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

    # TODO delete what isn't used
    xi = system_solver.xi
    yi = system_solver.yi
    zi = system_solver.zi
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

    bxGHbz = system_solver.bxGHbz
    Q1x = system_solver.Q1x
    GQ1x = system_solver.GQ1x
    HGQ1x = system_solver.HGQ1x
    GHGQ1x = system_solver.GHGQ1x
    Q2div = system_solver.Q2div

    GQ1 = system_solver.GQ1
    GQ2 = system_solver.GQ2
    HGQ2 = system_solver.HGQ2
    Q2GHGQ2 = system_solver.Q2GHGQ2

    Q2x = system_solver.Q2x
    Gxi = system_solver.Gxi
    HGxi = system_solver.HGxi
    GHGxi = system_solver.GHGxi

    HGQ1x_k = system_solver.HGQ1x_k
    GQ1x_k = system_solver.GQ1x_k
    HGQ2_k = system_solver.HGQ2_k
    GQ2_k = system_solver.GQ2_k
    HGxi_k = system_solver.HGxi_k
    Gxi_k = system_solver.Gxi_k

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
        grad_k = Cones.grad(cone_k)
        if Cones.use_dual(cone_k)
            @. z1_temp_k[k] /= mu
            @. z2_temp_k[k] = (duals_k + z2_temp_k[k]) / mu
            @. z3_temp_k[k] = duals_k / mu + grad_k
            Cones.inv_hess_prod!(z_k[k], z_temp_k[k], cone_k)
        else
            @. z1_temp_k[k] *= mu
            @. z2_temp_k[k] *= mu
            Cones.hess_prod!(z1_k[k], z1_temp_k[k], cone_k)
            Cones.hess_prod!(z2_k[k], z2_temp_k[k], cone_k)
            @. z2_k[k] += duals_k
            @. z3_k[k] = duals_k + grad_k * mu
        end
    end

    # TODO maybe replace with Diagonal(blocks) matrix product
    function block_hessian_product!(prod_k, arr_k)
        for k in eachindex(cones)
            cone_k = cones[k]
            if Cones.use_dual(cone_k)
                Cones.inv_hess_prod!(prod_k[k], arr_k[k], cone_k)
                prod_k[k] ./= mu
            else
                Cones.hess_prod!(prod_k[k], arr_k[k], cone_k)
                prod_k[k] .*= mu
            end
        end
    end

    # TODO cleanup all below

    # Q1x = Q1*R'\by
    # ldiv!(model.Ap_R', yi)
    # mul!(Q1x, model.Ap_Q1, yi)
    ldiv!(model.Ap_R', yi)

    # bxGHbz = bx + G'*Hbz
    mul!(bxGHbz, model.G', zi)
    @. bxGHbz += xi

    # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
    # mul!(GQ1x, model.G, Q1x)
    # block_hessian_product!(HGQ1x_k, GQ1x_k)
    # mul!(GHGQ1x, model.G', HGQ1x)
    # @. GHGQ1x = bxGHbz - GHGQ1x
    # mul!(Q2div, model.Ap_Q2', GHGQ1x)

    # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
    mul!(GQ1x, GQ1, yi)
    block_hessian_product!(HGQ1x_k, GQ1x_k)
    QpbxGHbz = model.Ap_Q' * bxGHbz # TODO allocs
    Q2div = QpbxGHbz[(model.p + 1):end, :] - GQ2' * HGQ1x # TODO allocs

    if !iszero(size(Q2div, 1))
        block_hessian_product!(HGQ2_k, GQ2_k)
        mul!(Q2GHGQ2, GQ2', HGQ2)

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
            F = hyp_chol!(Symmetric(Q2GHGQ2)) # TODO prealloc blasreal cholesky auxiliary vectors using posvx
            if !isposdef(F)
                println("dense linear system matrix factorization failed")
                mul!(Q2GHGQ2, GQ2', HGQ2)
                Q2GHGQ2 += T(1e-4) * I
                if T <: BlasReal
                    F = bunchkaufman!(Symmetric(Q2GHGQ2), true, check = false) # TODO prealloc with old sysvx code; not implemented for generic reals
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
            ldiv!(F, Q2div)
        end
    end

    # mul!(Q2x, model.Ap_Q2, Q2div)

    # xi = Q1x + Q2x
    # @. xi = Q1x + Q2x

    # xi = Q1x + Q2x
    xi .= model.Ap_Q * vcat(yi, Q2div) # TODO allocs

    # yi = Ri*Q1'*(bxGHbz - GHG*xi)
    # mul!(Gxi, model.G, xi)
    # block_hessian_product!(HGxi_k, Gxi_k)
    # mul!(GHGxi, model.G', HGxi)
    # @. bxGHbz -= GHGxi
    # mul!(yi, model.Ap_Q1', bxGHbz)
    # ldiv!(model.Ap_R, yi)

    # yi = Ri*Q1'*(bxGHbz - GHG*xi)
    mul!(Gxi, model.G, xi)
    block_hessian_product!(HGxi_k, Gxi_k)
    yi .= QpbxGHbz[1:model.p, :] - GQ1' * HGxi # TODO allocs
    ldiv!(model.Ap_R, yi)

    # zi = HG*xi - Hbz
    @. zi = HGxi - zi

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



# TODO singular recovery using A'*A

# Q2GHGQ2_fact = cholesky!(Q2GHGQ2, Val(true), check = false)
# singular = !isposdef(Q2GHGQ2_fact)
#
# if singular
#     println("singular Q2GHGQ2")
#
#     Q2GHGQ2 = Symmetric(model.Ap_Q2' * (GHG + model.A' * model.A + 1e-4I) * model.Ap_Q2)
#     # @show eigvals(Q2GHGQ2)
#     Q2GHGQ2_fact = cholesky!(Q2GHGQ2, Val(true), check = false)
#     if !isposdef(Q2GHGQ2_fact)
#         error("could not fix singular Q2GHGQ2")
#     end
# end
#
# xGHz = xi + model.G' * zi
# if singular
#     xGHz += model.A' * yi # TODO should this be minus
# end
