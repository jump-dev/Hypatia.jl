
mutable struct QRCholCombinedHSDSystemSolver <: CombinedHSDSystemSolver
    Ap_RiQ1t # TODO maybe do this lazily

    xi::Matrix{Float64}
    yi::Matrix{Float64}
    zi::Matrix{Float64}

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
    z_temp_k
    z1_temp_k
    z2_temp_k
    z3_temp_k

    G_k
    HG::Matrix{Float64}
    HG_k

    function QRCholCombinedHSDSystemSolver(model::Models.PreprocessedLinearModel)
        (n, p, q) = (model.n, model.p, model.q)
        system_solver = new()

        system_solver.Ap_RiQ1t = model.Ap_R \ (model.Ap_Q1')

        xi = Matrix{Float64}(undef, n, 3)
        yi = Matrix{Float64}(undef, p, 3)
        zi = Matrix{Float64}(undef, q, 3)
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
        system_solver.z_temp_k = [view(zi_temp, idxs, :) for idxs in model.cone_idxs]
        system_solver.z1_temp_k = [view(zi_temp, idxs, 1) for idxs in model.cone_idxs]
        system_solver.z2_temp_k = [view(zi_temp, idxs, 2) for idxs in model.cone_idxs]
        system_solver.z3_temp_k = [view(zi_temp, idxs, 3) for idxs in model.cone_idxs]

        system_solver.G_k = [view(model.G, idxs, :) for idxs in model.cone_idxs]
        system_solver.HG = similar(model.G)
        system_solver.HG_k = [view(system_solver.HG, idxs, :) for idxs in model.cone_idxs]

        return system_solver
    end
end

function get_combined_directions(solver::HSDSolver, system_solver::QRCholCombinedHSDSystemSolver)
    model = solver.model
    cones = model.cones
    cone_idxs = model.cone_idxs

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
    z_temp_k = system_solver.z_temp_k
    z1_temp_k = system_solver.z1_temp_k
    z2_temp_k = system_solver.z2_temp_k
    z3_temp_k = system_solver.z3_temp_k

    G_k = system_solver.G_k
    HG = system_solver.HG
    HG_k = system_solver.HG_k

    @timeit "setup xy" begin
    @. x1 = -model.c
    @. x2 = solver.x_residual
    @. x3 = 0.0
    @. y1 = model.b
    @. y2 = -solver.y_residual
    @. y3 = 0.0
    end

    @timeit "setup z" begin
    @. z1_temp = model.h
    @. z2_temp = -solver.z_residual
    for k in eachindex(cones)
        cone_k = cones[k]
        idxs = cone_idxs[k]
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        if Cones.use_dual(cone_k)
            @. z1_temp_k[k] /= solver.mu
            @. z2_temp_k[k] = (duals_k + z2_temp_k[k]) / solver.mu
            @. z3_temp_k[k] = duals_k / solver.mu + grad_k
            ldiv!(z_k[k], Cones.hess_fact(cone_k), z_temp_k[k])
        else
            @. z1_temp_k[k] *= solver.mu
            @. z2_temp_k[k] *= solver.mu
            mul!(z1_k[k], Cones.hess(cone_k), z1_temp_k[k])
            mul!(z2_k[k], Cones.hess(cone_k), z2_temp_k[k])
            @. z2_k[k] += duals_k
            @. z3_k[k] = duals_k + grad_k * solver.mu
        end
    end
    end




    # TODO multiply G by Ap_Q2 first, but then need to use hessian inverse etc later instead of HG or GHG
    @timeit "HG" begin
    for k in eachindex(cones)
        cone_k = cones[k]
        idxs = cone_idxs[k]
        if Cones.use_dual(cone_k)
            ldiv!(HG_k[k], Cones.hess_fact(cone_k), G_k[k])
            @. HG_k[k] /= solver.mu
        else
            mul!(HG_k[k], Cones.hess(cone_k), G_k[k])
            @. HG_k[k] *= solver.mu
        end
    end
    end

    # TODO remove allocs here
    @timeit "GHG" begin
    GHG = Symmetric(model.G' * HG)
    Q2GHGQ2 = Symmetric(model.Ap_Q2' * GHG * model.Ap_Q2)
    end

    @timeit "fact" begin
    Q2GHGQ2_fact = cholesky!(Q2GHGQ2, Val(true), check = false)
    singular = !isposdef(Q2GHGQ2_fact)
    # Q2GHGQ2_fact = bunchkaufman!(Q2GHGQ2, true, check = false)
    # singular = !issuccess(Q2GHGQ2_fact)

    if singular
        println("singular Q2GHGQ2")

        # Q2GHGQ2 = Symmetric(model.Ap_Q2' * (GHG + model.A' * model.A + 1e-4I) * model.Ap_Q2)
        # # @show eigvals(Q2GHGQ2)
        # Q2GHGQ2_fact = cholesky!(Q2GHGQ2, Val(true), check = false)
        # if !isposdef(Q2GHGQ2_fact)
        #     error("could not fix singular Q2GHGQ2")
        # end

        Q2GHGQ2 = Symmetric(model.Ap_Q2' * GHG * model.Ap_Q2 + 1e-4 * I)
        Q2GHGQ2_fact = bunchkaufman!(Q2GHGQ2, true, check = false)
        if !issuccess(Q2GHGQ2_fact)
            error("could not fix singular Q2GHGQ2")
        end
    end
    end

    @timeit "post fact" begin
    xGHz = xi + model.G' * zi
    # if singular
    #     xGHz += model.A' * yi # TODO should this be minus
    # end

    x = system_solver.Ap_RiQ1t' * yi
    Q2div = model.Ap_Q2' * (xGHz - GHG * x)
    ldiv!(Q2GHGQ2_fact, Q2div)
    x += model.Ap_Q2 * Q2div

    y = system_solver.Ap_RiQ1t * (xGHz - GHG * x)

    z = HG * x - zi
    end

    # combine
    @timeit "lift" begin
    x1 = view(x, :, 1)
    y1 = view(y, :, 1)
    z1 = view(z, :, 1)
    x23 = view(x, :, 2:3)
    y23 = view(y, :, 2:3)
    z23 = view(z, :, 2:3)

    tau_rhs = [(solver.kap + solver.primal_obj_t - solver.dual_obj_t)  0.0]
    kap_rhs = [-solver.kap  (-solver.kap + solver.mu / solver.tau)]
    tau_denom = solver.mu / solver.tau / solver.tau - dot(model.c, x1) - dot(model.b, y1) - dot(model.h, z1)
    tau_sol = (tau_rhs + kap_rhs + model.c' * x23 + model.b' * y23 + model.h' * z23) ./ tau_denom

    x_sol = x23 + x1 * tau_sol
    y_sol = y23 + y1 * tau_sol
    z_sol = z23 + z1 * tau_sol

    s_sol = -model.G * x_sol + model.h * tau_sol - [solver.z_residual  zeros(model.q)]

    kap_sol = -model.c' * x_sol - model.b' * y_sol - model.h' * z_sol - tau_rhs
    end

    return (x_sol, y_sol, z_sol, s_sol, tau_sol, kap_sol)
end



# #=
# Copyright 2018, Chris Coey and contributors
#
# solve two symmetric linear systems and combine solutions (inspired by CVXOPT)
# QR plus either Cholesky factorization or iterative conjugate gradients method
# (1) eliminate equality constraints via QR of A'
# (2) solve reduced symmetric system by Cholesky or iterative method
# =#
#
# mutable struct QRSymm <: LinearSystemSolver
#     # TODO can remove some of the prealloced arrays after github.com/JuliaLang/julia/issues/23919 is resolved
#     useiterative
#     userefine
#
#     cone
#     c
#     b
#     G
#     h
#     Q2
#     Ap_RiQ1t
#
#     bxGHbz
#     Q1x
#     GQ1x
#     HGQ1x
#     GHGQ1x
#     Q2div
#     GQ2
#     HGQ2
#     Q2GHGQ2
#     Q2x
#     Gxi
#     HGxi
#     GHGxi
#
#     zi
#     yi
#     xi
#
#     Q2divcopy
#     lsferr
#     lsberr
#     lswork
#     lsiwork
#     lsAF
#     lsS
#     ipiv
#
#     # cgstate
#     # lprecond
#     # Q2sol
#
#     function QRSymm(
#         c::Vector{Float64},
#         A::AbstractMatrix{Float64},
#         b::Vector{Float64},
#         G::AbstractMatrix{Float64},
#         h::Vector{Float64},
#         cone::Cones.Cone,
#         Q2::AbstractMatrix{Float64},
#         Ap_RiQ1t::AbstractMatrix{Float64};
#         useiterative::Bool = false,
#         userefine::Bool = false,
#         )
#         @assert !useiterative # TODO disabled for now
#
#         L = new()
#         (n, p, q) = (length(c), length(b), length(h))
#         nmp = n - p
#
#         L.useiterative = useiterative
#         L.userefine = userefine
#
#         L.cone = cone
#         L.c = c
#         L.b = b
#         L.G = G
#         L.h = h
#         L.Q2 = Q2
#         L.Ap_RiQ1t = Ap_RiQ1t
#
#         L.bxGHbz = Matrix{Float64}(undef, n, 2)
#         L.Q1x = similar(L.bxGHbz)
#         L.GQ1x = Matrix{Float64}(undef, q, 2)
#         L.HGQ1x = similar(L.GQ1x)
#         L.GHGQ1x = Matrix{Float64}(undef, n, 2)
#         L.Q2div = Matrix{Float64}(undef, nmp, 2)
#         L.GQ2 = G*Q2
#         L.HGQ2 = similar(L.GQ2)
#         L.Q2GHGQ2 = Matrix{Float64}(undef, nmp, nmp)
#         L.Q2x = similar(L.Q1x)
#         L.Gxi = similar(L.GQ1x)
#         L.HGxi = similar(L.Gxi)
#         L.GHGxi = similar(L.GHGQ1x)
#
#         L.zi = Matrix{Float64}(undef, q, 2)
#         L.yi = Matrix{Float64}(undef, p, 2)
#         L.xi = Matrix{Float64}(undef, n, 2)
#
#         # for linear system solve with refining
#         L.Q2divcopy = similar(L.Q2div)
#         L.lsferr = Vector{Float64}(undef, 2)
#         L.lsberr = Vector{Float64}(undef, 2)
#         L.lsAF = Matrix{Float64}(undef, nmp, nmp)
#         # sysvx
#         L.lswork = Vector{Float64}(undef, 5*nmp)
#         L.lsiwork = Vector{BlasInt}(undef, nmp)
#         L.ipiv = Vector{BlasInt}(undef, nmp)
#         # # posvx
#         # L.lswork = Vector{Float64}(undef, 3*nmp)
#         # L.lsiwork = Vector{BlasInt}(undef, nmp)
#         # L.lsS = Vector{Float64}(undef, nmp)
#
#         # # for iterative only
#         # if useiterative
#         #     cgu = zeros(nmp)
#         #     L.cgstate = IterativeSolvers.CGStateVariables{Float64, Vector{Float64}}(cgu, similar(cgu), similar(cgu))
#         #     L.lprecond = IterativeSolvers.Identity()
#         #     L.Q2sol = zeros(nmp)
#         # end
#
#         return L
#     end
# end
#
#
# QRSymm(
#     c::Vector{Float64},
#     A::AbstractMatrix{Float64},
#     b::Vector{Float64},
#     G::AbstractMatrix{Float64},
#     h::Vector{Float64},
#     cone::Cones.Cone;
#     useiterative::Bool = false,
#     userefine::Bool = false,
#     ) = error("to use a QRSymm for linear system solves, the data must be preprocessed and Q2 and Ap_RiQ1t must be passed into the QRSymm constructor")
#
#
# # solve two symmetric systems and combine the solutions for x, y, z, s, kap, tau
# # TODO update math description
# # TODO use in-place mul-add when available in Julia, see https://github.com/JuliaLang/julia/issues/23919
# function solvelinsys6!(
#     rhs_tx::Vector{Float64},
#     rhs_ty::Vector{Float64},
#     rhs_tz::Vector{Float64},
#     rhs_kap::Float64,
#     rhs_ts::Vector{Float64},
#     rhs_tau::Float64,
#     mu::Float64,
#     tau::Float64,
#     L::QRSymm,
#     )
#     (zi, yi, xi) = (L.zi, L.yi, L.xi)
#     @. yi[:, 1] = L.b
#     @. yi[:, 2] = -rhs_ty
#     @. xi[:, 1] = -L.c
#     @. xi[:, 2] = rhs_tx
#     z1 = view(zi, :, 1)
#     z2 = view(zi, :, 2)
#
#     # calculate z2
#     @. z2 = -rhs_tz
#     for k in eachindex(L.cones)
#         a1k = view(z1, L.cone_idxs[k])
#         a2k = view(z2, L.cone_idxs[k])
#         a3k = view(rhs_ts, L.cone_idxs[k])
#         if L.Cones.use_dual(cone_k)
#             @. a1k = a2k - a3k
#             Cones.calcHiarr!(a2k, a1k, L.cones[k])
#             a2k ./= mu
#         elseif !iszero(a3k) # TODO rhs_ts = 0 for correction steps, so can just check if doing correction
#             Cones.calcHarr!(a1k, a3k, L.cones[k])
#             @. a2k -= mu * a1k
#         end
#     end
#
#     # calculate z1
#     if iszero(L.h) # TODO can check once when creating cache
#         z1 .= 0.0
#     else
#         for k in eachindex(L.cones)
#             a1k = view(L.h, L.cone_idxs[k])
#             a2k = view(z1, L.cone_idxs[k])
#             if L.Cones.use_dual(cone_k)
#                 Cones.calcHiarr!(a2k, a1k, L.cones[k])
#                 a2k ./= mu
#             else
#                 Cones.calcHarr!(a2k, a1k, L.cones[k])
#                 a2k .*= mu
#             end
#         end
#     end
#
#     # bxGHbz = bx + G'*Hbz
#     mul!(L.bxGHbz, L.G', zi)
#     @. L.bxGHbz += xi
#
#     # Q1x = Q1*Ri'*by
#     mul!(L.Q1x, L.Ap_RiQ1t', yi)
#
#     # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
#     mul!(L.GQ1x, L.G, L.Q1x)
#     for k in eachindex(L.cones)
#         a1k = view(L.GQ1x, L.cone_idxs[k], :)
#         a2k = view(L.HGQ1x, L.cone_idxs[k], :)
#         if L.Cones.use_dual(cone_k)
#             Cones.calcHiarr!(a2k, a1k, L.cones[k])
#             a2k ./= mu
#         else
#             Cones.calcHarr!(a2k, a1k, L.cones[k])
#             a2k .*= mu
#         end
#     end
#     mul!(L.GHGQ1x, L.G', L.HGQ1x)
#     @. L.GHGQ1x = L.bxGHbz - L.GHGQ1x
#     mul!(L.Q2div, L.Q2', L.GHGQ1x)
#
#     if size(L.Q2div, 1) > 0
#         for k in eachindex(L.cones)
#             a1k = view(L.GQ2, L.cone_idxs[k], :)
#             a2k = view(L.HGQ2, L.cone_idxs[k], :)
#             if L.Cones.use_dual(cone_k)
#                 Cones.calcHiarr!(a2k, a1k, L.cones[k])
#                 a2k ./= mu
#             else
#                 Cones.calcHarr!(a2k, a1k, L.cones[k])
#                 a2k .*= mu
#             end
#         end
#         mul!(L.Q2GHGQ2, L.GQ2', L.HGQ2)
#
#         # F = bunchkaufman!(Symmetric(L.Q2GHGQ2), true, check = false)
#         # if !issuccess(F)
#         #     println("linear system matrix factorization failed")
#         #     mul!(L.Q2GHGQ2, L.GQ2', L.HGQ2)
#         #     L.Q2GHGQ2 += 1e-6I
#         #     F = bunchkaufman!(Symmetric(L.Q2GHGQ2), true, check = false)
#         #     if !issuccess(F)
#         #         error("could not fix failure of positive definiteness; terminating")
#         #     end
#         # end
#         # ldiv!(F, L.Q2div)
#
#         success = hypatia_sysvx!(L.Q2divcopy, L.Q2GHGQ2, L.Q2div, L.lsferr, L.lsberr, L.lswork, L.lsiwork, L.lsAF, L.ipiv)
#         if !success
#             # println("linear system matrix factorization failed")
#             mul!(L.Q2GHGQ2, L.GQ2', L.HGQ2)
#             L.Q2GHGQ2 += 1e-4I
#             mul!(L.Q2div, L.Q2', L.GHGQ1x)
#             success = hypatia_sysvx!(L.Q2divcopy, L.Q2GHGQ2, L.Q2div, L.lsferr, L.lsberr, L.lswork, L.lsiwork, L.lsAF, L.ipiv)
#             if !success
#                 error("could not fix linear system solve failure; terminating")
#             end
#         end
#         L.Q2div .= L.Q2divcopy
#     end
#     mul!(L.Q2x, L.Q2, L.Q2div)
#
#     # xi = Q1x + Q2x
#     @. xi = L.Q1x + L.Q2x
#
#     # yi = Ri*Q1'*(bxGHbz - GHG*xi)
#     mul!(L.Gxi, L.G, xi)
#     for k in eachindex(L.cones)
#         a1k = view(L.Gxi, L.cone_idxs[k], :)
#         a2k = view(L.HGxi, L.cone_idxs[k], :)
#         if L.Cones.use_dual(cone_k)
#             Cones.calcHiarr!(a2k, a1k, L.cones[k])
#             a2k ./= mu
#         else
#             Cones.calcHarr!(a2k, a1k, L.cones[k])
#             a2k .*= mu
#         end
#     end
#     mul!(L.GHGxi, L.G', L.HGxi)
#     @. L.bxGHbz -= L.GHGxi
#     mul!(yi, L.Ap_RiQ1t, L.bxGHbz)
#
#     # zi = HG*xi - Hbz
#     @. zi = L.HGxi - zi
#
#     # combine
#     @views dir_tau = (rhs_tau + rhs_kap + dot(L.c, xi[:, 2]) + dot(L.b, yi[:, 2]) + dot(L.h, z2)) /
#         (mu / tau / tau - dot(L.c, xi[:, 1]) - dot(L.b, yi[:, 1]) - dot(L.h, z1))
#     @. @views rhs_tx = xi[:, 2] + dir_tau * xi[:, 1]
#     @. @views rhs_ty = yi[:, 2] + dir_tau * yi[:, 1]
#     @. rhs_tz = z2 + dir_tau * z1
#     mul!(z1, L.G, rhs_tx)
#     @. rhs_ts = -z1 + L.h * dir_tau - rhs_ts
#     dir_kap = -dot(L.c, rhs_tx) - dot(L.b, rhs_ty) - dot(L.h, rhs_tz) - rhs_tau
#
#     return (dir_kap, dir_tau)
# end
#
#
#
# #=
# Copyright 2018, Chris Coey and contributors
#
# eliminates the s row and column from the 4x4 system and performs one 3x3 linear system solve (see naive3 method)
# requires QR-based preprocessing of A', uses resulting Q2 and Ap_RiQ1t matrices to eliminate equality constraints
# uses a Cholesky to solve a reduced symmetric linear system
#
# TODO option for solving linear system with positive definite matrix using iterative method (eg CG)
# TODO refactor many common elements with chol2
# =#
#
# mutable struct QRChol <: LinearSystemSolver
#     n
#     p
#     q
#     P
#     A
#     G
#     Q2
#     Ap_RiQ1t
#     cone
#
#     function QRChol(
#         P::AbstractMatrix{Float64},
#         c::Vector{Float64},
#         A::AbstractMatrix{Float64},
#         b::Vector{Float64},
#         G::AbstractMatrix{Float64},
#         h::Vector{Float64},
#         cone::Cone,
#         Q2::AbstractMatrix{Float64},
#         Ap_RiQ1t::AbstractMatrix{Float64},
#         )
#         L = new()
#         (n, p, q) = (length(c), length(b), length(h))
#         (L.n, L.p, L.q, L.P, L.A, L.G, L.Q2, L.Ap_RiQ1t, L.cone) = (n, p, q, P, A, G, Q2, Ap_RiQ1t, cone)
#         return L
#     end
# end
#
# QRChol(
#     c::Vector{Float64},
#     A::AbstractMatrix{Float64},
#     b::Vector{Float64},
#     G::AbstractMatrix{Float64},
#     h::Vector{Float64},
#     cone::Cone,
#     Q2::AbstractMatrix{Float64},
#     Ap_RiQ1t::AbstractMatrix{Float64},
#     ) = QRChol(Symmetric(spzeros(length(c), length(c))), c, A, b, G, h, cone, Q2, Ap_RiQ1t)
#
#
# # solve system for x, y, z, s
# function solvelinsys4!(
#     xrhs::Vector{Float64},
#     yrhs::Vector{Float64},
#     zrhs::Vector{Float64},
#     srhs::Vector{Float64},
#     mu::Float64,
#     L::QRChol,
#     )
#     (n, p, q, P, A, G, Q2, Ap_RiQ1t, cone) = (L.n, L.p, L.q, L.P, L.A, L.G, L.Q2, L.Ap_RiQ1t, L.cone)
#
#     # TODO refactor the conversion to 3x3 system and back (start and end)
#     zrhs3 = copy(zrhs)
#     for k in eachindex(cones)
#         sview = view(srhs, cone_idxs[k])
#         zview = view(zrhs3, cone_idxs[k])
#         if Cones.use_dual(cone_k) # G*x - mu*H*z = zrhs - srhs
#             zview .-= sview
#         else # G*x - (mu*H)\z = zrhs - (mu*H)\srhs
#             calcHiarr!(sview, cones[k])
#             @. zview -= sview / mu
#         end
#     end
#
#     HG = Matrix{Float64}(undef, q, n)
#     for k in eachindex(cones)
#         Gview = view(G, cone_idxs[k], :)
#         HGview = view(HG, cone_idxs[k], :)
#         if Cones.use_dual(cone_k)
#             calcHiarr!(HGview, Gview, cones[k])
#             HGview ./= mu
#         else
#             calcHarr!(HGview, Gview, cones[k])
#             HGview .*= mu
#         end
#     end
#     GHG = Symmetric(G' * HG)
#     GHG = Symmetric(P + GHG)
#     Q2GHGQ2 = Symmetric(Q2' * GHG * Q2)
#     F = cholesky!(Q2GHGQ2, Val(true), check = false)
#     singular = !isposdef(F)
#     # F = bunchkaufman!(Q2GHGQ2, true, check = false)
#     # singular = !issuccess(F)
#
#     if singular
#         println("singular Q2GHGQ2")
#         Q2GHGQ2 = Symmetric(Q2' * (GHG + A' * A) * Q2)
#         # @show eigvals(Q2GHGQ2)
#         F = cholesky!(Q2GHGQ2, Val(true), check = false)
#         if !isposdef(F)
#             error("could not fix singular Q2GHGQ2")
#         end
#         # F = bunchkaufman!(Q2GHGQ2, true, check = false)
#         # if !issuccess(F)
#         #     error("could not fix singular Q2GHGQ2")
#         # end
#     end
#
#     Hz = similar(zrhs3)
#     for k in eachindex(cones)
#         zview = view(zrhs3, cone_idxs[k], :)
#         Hzview = view(Hz, cone_idxs[k], :)
#         if Cones.use_dual(cone_k)
#             calcHiarr!(Hzview, zview, cones[k])
#             Hzview ./= mu
#         else
#             calcHarr!(Hzview, zview, cones[k])
#             Hzview .*= mu
#         end
#     end
#     xGHz = xrhs + G' * Hz
#     if singular
#         xGHz += A' * yrhs # TODO should this be minus
#     end
#
#     x = Ap_RiQ1t' * yrhs
#     Q2div = Q2' * (xGHz - GHG * x)
#     ldiv!(F, Q2div)
#     x += Q2 * Q2div
#
#     y = Ap_RiQ1t * (xGHz - GHG * x)
#
#     z = similar(zrhs3)
#     Gxz = G * x - zrhs3
#     for k in eachindex(cones)
#         Gxzview = view(Gxz, cone_idxs[k], :)
#         zview = view(z, cone_idxs[k], :)
#         if Cones.use_dual(cone_k)
#             calcHiarr!(zview, Gxzview, cones[k])
#             zview ./= mu
#         else
#             calcHarr!(zview, Gxzview, cones[k])
#             zview .*= mu
#         end
#     end
#
#     srhs .= zrhs # G*x + s = zrhs
#     xrhs .= x
#     yrhs .= y
#     zrhs .= z
#     srhs .-= G * x1
#
#     return
# end
