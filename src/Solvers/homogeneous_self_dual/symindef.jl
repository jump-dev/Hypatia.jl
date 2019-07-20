#=
Copyright 2018, Chris Coey and contributors

symmetric-indefinite linear system solver
solves linear system in naive.jl via the following procedure

A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
(pr bar) -G_k*x + (mu*H_k)^-1*z_k + h_k*tau = zrhs_k + (mu*H_k)^-1*srhs_k
(du bar) -G_k*x + mu*H_k*z_k + h_k*tau = zrhs_k + srhs_k
-c'x - b'y - h'z + mu/(taubar^2)*tau = taurhs + kaprhs

optionally avoid use of inverse Hessians, by replacing the (pr bar) constraints with
(pr bar) -mu*H_k*G_k*x + z_k + mu*H_k*h_k*tau = mu*H_k*zrhs_k + srhs_k
i.e. premultiplying by mu*H_k

eliminate tau via a procedure similar to that described by S7.4 of
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
and symmetrize the LHS matrix by multiplying some equations by -1

TODO reduce allocations
=#

using LinearAlgebra.LAPACK: BlasInt, chklapackerror, @blasfunc, liblapack
using LinearAlgebra.LAPACK: checksquare

mutable struct SymIndefCombinedHSDSystemSolver{T <: HypReal} <: CombinedHSDSystemSolver{T}
    use_sparse::Bool
    use_hess_inv::Bool
    use_iterative::Bool

    lhs_copy
    lhs
    rhs::Matrix{T}
    prevsol::Matrix{T}

    x1
    x2
    x3
    y1
    y2
    y3
    z1
    z2
    z3
    z1_k
    z2_k
    z3_k
    s1::Vector{T}
    s2::Vector{T}
    s3::Vector{T}
    s1_k
    s2_k
    s3_k

    function SymIndefCombinedHSDSystemSolver{T}(model::Models.LinearModel{T}; use_sparse::Bool = false, use_hess_inv::Bool = false, use_iterative::Bool = false) where {T <: HypReal}
        (n, p, q) = (model.n, model.p, model.q)
        npq = n + p + q
        system_solver = new{T}()
        system_solver.use_sparse = use_sparse
        system_solver.use_hess_inv = use_hess_inv
        system_solver.use_iterative = use_iterative

        # x y z
        # symmetric, lower triangle filled only
        if use_sparse
            system_solver.lhs_copy = T[
                spzeros(T,n,n)  spzeros(T,n,p)  spzeros(T,n,q);
                model.A         spzeros(T,p,p)  spzeros(T,p,q);
                model.G         spzeros(T,q,p)  sparse(-one(T)*I,q,q);
            ]
            @assert issparse(system_solver.lhs_copy)
        else
            system_solver.lhs_copy = T[
                zeros(T,n,n)  zeros(T,n,p)  zeros(T,n,q);
                model.A       zeros(T,p,p)  zeros(T,p,q);
                model.G       zeros(T,q,p)  Matrix(-one(T)*I,q,q);
            ]
        end

        system_solver.lhs = similar(system_solver.lhs_copy)

        rhs = Matrix{T}(undef, npq, 3)
        system_solver.rhs = rhs
        system_solver.prevsol = zeros(T, npq, 3)
        rows = 1:n
        system_solver.x1 = view(rhs, rows, 1)
        system_solver.x2 = view(rhs, rows, 2)
        system_solver.x3 = view(rhs, rows, 3)
        rows = (n + 1):(n + p)
        system_solver.y1 = view(rhs, rows, 1)
        system_solver.y2 = view(rhs, rows, 2)
        system_solver.y3 = view(rhs, rows, 3)
        rows = (n + p + 1):(n + p + q)
        system_solver.z1 = view(rhs, rows, 1)
        system_solver.z2 = view(rhs, rows, 2)
        system_solver.z3 = view(rhs, rows, 3)
        z_start = n + p
        system_solver.z1_k = [view(rhs, z_start .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
        system_solver.z2_k = [view(rhs, z_start .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]
        system_solver.z3_k = [view(rhs, z_start .+ model.cone_idxs[k], 3) for k in eachindex(model.cones)]
        system_solver.s1 = similar(rhs, q)
        system_solver.s2 = similar(rhs, q)
        system_solver.s3 = similar(rhs, q)

        return system_solver
    end
end

for (syequb_, elty, relty) in
    ((:dsyequb_, :Float64, :Float64),
     # (:syequb_, :ComplexF64, :Float64),
     # (:syequb_, :ComplexF32, :Float32),
     # (:syequb_, :Float32, :Float32),
     )
    @eval begin
        function syequb(A::AbstractMatrix{$elty})
            m,n = size(A)
            lda = max(1, stride(A,2))
            S = Vector{$relty}(undef, n)
            info = Ref{BlasInt}()
            scond = Ref{$relty}()
            amax = Ref{$relty}()
            work = Vector{$relty}(undef, 2 * n)
            ccall((@blasfunc($syequb_), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$relty},
                   Ptr{$relty}, Ptr{$relty}, Ptr{$relty},
                   Ptr{BlasInt}),
                  'L', n, A, lda, S, scond, amax, work, info)
            chklapackerror(info[])
            S, scond, amax
        end
    end
end

function equilibrators(A::AbstractMatrix{T}) where {T}
    abs1(x::Real) = abs(x)
    abs1(x::Complex) = abs(real(x)) + abs(imag(x))
    B = abs1.(A)
    m,n = size(B)
    R = zeros(T,m)
    C = zeros(T,n)
    @inbounds for j=1:n
        R .= max.(R,view(B,:,j))
    end
    @inbounds for i=1:m
        if R[i] > 0
            R[i] = T(2)^floor(Int,log2(R[i]))
        end
    end
    # R .= 1 ./ R
    @inbounds for i=1:m
        C .= max.(C,view(B,i,:) / R[i])
    end
    @inbounds for j=1:n
        if C[j] > 0
            C[j] = T(2)^floor(Int,log2(C[j]))
        end
    end
    # C .= 1 ./ C
    R,C
end

function maxels(A)
    maxels = maximum(A, dims = 2)
    maxels = map(x -> iszero(x) ? 1.0 : x, maxels)
    pl = Diagonal(maxels[:])
    maxels = maximum(A, dims = 1)
    maxels = map(x -> iszero(x) ? 1.0 : x, maxels)
    pr = Diagonal(maxels[:])
    return (pl, pr)
end

function get_combined_directions(solver::HSDSolver{T}, system_solver::SymIndefCombinedHSDSystemSolver{T}) where {T <: HypReal}
    use_hess_inv = system_solver.use_hess_inv
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    mu = solver.mu
    tau = solver.tau
    kap = solver.kap

    x1 = system_solver.x1
    x2 = system_solver.x2
    x3 = system_solver.x3
    y1 = system_solver.y1
    y2 = system_solver.y2
    y3 = system_solver.y3
    z1 = system_solver.z1
    z2 = system_solver.z2
    z3 = system_solver.z3
    z1_k = system_solver.z1_k
    z2_k = system_solver.z2_k
    z3_k = system_solver.z3_k
    s1 = system_solver.s1
    s2 = system_solver.s2
    s3 = system_solver.s3

    @. x1 = -model.c
    x2 .= solver.x_residual
    x3 .= zero(T)
    y1 .= model.b
    @. y2 = -solver.y_residual
    y3 .= zero(T)
    z1 .= model.h
    @. z2 .= -solver.z_residual
    z3 .= zero(T)

    # update lhs matrix
    copyto!(lhs, system_solver.lhs_copy)
    for k in eachindex(cones)
        cone_k = cones[k]
        idxs = model.cone_idxs[k]
        rows = (n + p) .+ idxs
        duals_k = solver.point.dual_views[k]
        g = Cones.grad(cone_k)
        if Cones.use_dual(cone_k)
            # G_k*x - mu*H_k*z_k = zrhs_k + srhs_k
            H = Cones.hess(cone_k)
            lhs[rows, rows] .= -mu * H
            @. z2_k[k] += duals_k
            @. z3_k[k] = duals_k + mu * g
        else
            if use_hess_inv
                # G_k*x - (mu*H_k)^-1*z_k = zrhs_k + (mu*H_k)^-1*srhs_k
                muHinv = Cones.inv_hess(cone_k) / mu
                lhs[rows, rows] .= -muHinv
                z2_k[k] .+= muHinv * duals_k
                z3_k[k] .= muHinv * (duals_k + mu * g)
            else
                # mu*H_k*G_k*x - z_k = mu*H_k*zrhs_k + srhs_k
                H = Cones.hess(cone_k)
                lhs[rows, 1:n] .= mu * H * model.G[idxs, :]
                lhs[rows, rows] .= -mu * H
                z1_k[k] .= mu * H * z1_k[k]
                z2_k[k] .= mu * H * z2_k[k]
                @. z2_k[k] += duals_k
                @. z3_k[k] = duals_k + mu * g
            end
        end
    end

    # solve system
    if system_solver.use_iterative

        ill_cond_block = lhs[(n + 1):end, (n + 1):end]
        AG = lhs[(n + 1):end, 1:n]
        W = I
        # preconditioner = [
        # W   zeros(n, p + q);
        # zeros(p + q, n)   ill_cond_block + AG * (W \ AG');
        # ]
        preconditioner = [
        I   zeros(n + p, q)
        zeros(q, n + p)    lhs[(n + p + 1):end, (n + p + 1):end]
        ]
        # @show cond(preconditioner \ Symmetric(LHS, :L))
        # preconditioner = I
        # d = diag(lhs)
        # preconditioner = Diagonal(map(di -> (iszero(di) ? 1.0 : di), d))
        # lhsp = preconditioner \ Symmetric(lhs, :L)

        # (R, C) = equilibrators(Symmetric(lhs, :L))
        # (pl, pr) = maxels(Symmetric(lhs, :L))
        # io = open("p1.csv", "a")

        for i in 1:3

            rhs2 = view(rhs, :, i)
            # rhsp = preconditioner \ rhs2

            prevsol = view(system_solver.prevsol, :, i)

            x0 = deepcopy(system_solver.prevsol)
            # @show norm(rhs2 - Symmetric(lhs, :L) * prevsol) < 1e-5

            # (prevsol, log) = IterativeSolvers.minres!(prevsol, Symmetric(lhs, :L), rhs2, maxiter = 1 * size(lhs, 2), reorth = true, log = true, tol = 1e-8) # goes badly
            # prevsol .= C * prevsol
            # @show size(lhs, 2), log.iters
            # @show log.isconverged
            # @show cond(Symmetric(lhs, :L))
            # @show cond(Diagonal(R) \ Symmetric(lhs, :L) / Diagonal(C))

            @timeit solver.timer "precond" S, scond, amax = syequb(lhs)
            S = Diagonal(S)
            # S = I

            # @show Symmetric(lhs, :L)
            # @show S * Symmetric(lhs, :L) * S
            # @show eigen(Symmetric(lhs, :L)).values
            # @show eigen(S * Symmetric(lhs, :L) * S).values
            # (x, log) = IterativeSolvers.gmres!(prevsol, Symmetric(S * Symmetric(lhs, :L) * S), S * rhs2, maxiter = 1 * size(lhs, 2), restart = size(lhs, 2), log = true, tol = 1e-12)
            prevsol .= S \ prevsol
            @timeit solver.timer "minres" (x, log) = IterativeSolvers.minres!(prevsol, Symmetric(S * Symmetric(lhs, :L) * S), S * rhs2, maxiter = size(lhs, 2), log = true, rtol = 1e-8, reorth = false)
            prevsol .= S * prevsol

            # print(io, "$(dot(x0[:, i], prevsol) / norm(x0[:, i]) / norm(prevsol)),")
            # print(io, "$(norm(x0[:, i]) / norm(prevsol)), $(norm(x0[:, i]) - norm(prevsol)),")
            # print(io, "$(norm(Symmetric(lhs, :L) \ rhs2 - x0[:, i])),$(norm(rhs2 - Symmetric(lhs, :L) * x0[:, i])), ,")

            # @timeit solver.timer "itersolve" (x, log) = IterativeSolvers.gmres!(prevsol, Symmetric(lhs, :L), rhs2, maxiter = 1 * size(lhs, 2), restart = div(size(lhs, 2), 1), log = true)

            # @show log.iters, size(lhs, 2)

            @show log.iters
            # prevsol = Symmetric(lhs, :L) \ rhs2
            # if !(log.isconverged)
            #     CSV.write("lhs.csv",  DataFrames.DataFrame(lhs), writeheader=false)
            #     CSV.write("rhs.csv",  DataFrames.DataFrame(rhs), writeheader=false)
            #     CSV.write("prevsol.csv",  DataFrames.DataFrame(x0), writeheader=false)
            #     error()
            # end
            rhs2 .= prevsol

        end
        # print(io, "\n")
        # close(io)

    else
        if system_solver.use_sparse
            F = ldlt(Symmetric(lhs, :L), check = false)
            if !issuccess(F)
                F = ldlt(Symmetric(lhs, :L), shift = 1e-6, check = true)
            end
            rhs .= F \ rhs
        else
            # CSV.write("lhs/lhs_$(round(Int, time() * 1000)).csv",  DataFrames.DataFrame(lhs), writeheader=false)
            # CSV.write("matrices/rhs_$(round(Int, time() * 1000)).csv",  DataFrames.DataFrame(rhs), writeheader=false)
            if T <: BlasReal
                F = bunchkaufman!(Symmetric(lhs, :L), true, check = true) # TODO doesn't work for generic reals (need LDLT)
                ldiv!(F, rhs)
            else
                rhs .= Symmetric(lhs, :L) \ rhs # TODO replace with a generic julia symmetric indefinite decomposition if available, see https://github.com/JuliaLang/julia/issues/10953
            end
            # @show size(rhs), size(lhs)
        end
        # F = lu!(lhs_symm)
        # ldiv!(F, rhs)
    end

    if !use_hess_inv
        for k in eachindex(cones)
            if !Cones.use_dual(cones[k])
                # z_k is premultiplied by (mu * H_k)^-1
                H = Cones.hess(cones[k])
                # TODO combine into one
                z1_k[k] .= mu * H * z1_k[k]
                z2_k[k] .= mu * H * z2_k[k]
                z3_k[k] .= mu * H * z3_k[k]
            end
        end
    end

    # lift to HSDE space
    # if not using inverse hessians, use modified h
    tau_denom = mu / tau / tau - dot(model.c, x1) - dot(model.b, y1) - dot(model.h, z1)

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

    (tau_pred, kap_pred) = lift!(x2, y2, z2, s2, kap + solver.primal_obj_t - solver.dual_obj_t, -kap)
    @. s2 -= solver.z_residual
    (tau_corr, kap_corr) = lift!(x3, y3, z3, s3, zero(T), -kap + mu / tau)

    return (x2, x3, y2, y3, z2, z3, s2, s3, tau_pred, tau_corr, kap_pred, kap_corr)
end
