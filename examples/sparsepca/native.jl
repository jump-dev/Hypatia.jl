#=
see "A Direct Formulation for Sparse PCA Using Semidefinite Programming" by
Alexandre d’Aspremont, Laurent El Ghaoui, Michael I. Jordan, Gert R. G. Lanckriet
=#

import Distributions
using SparseArrays

struct SparsePCANative{T <: Real} <: ExampleInstanceNative{T}
    p::Int
    k::Int
    use_epinorminfdual::Bool # use dual of epinorminf cone, else nonnegative cones
    noise_ratio::T
end

function build(inst::SparsePCANative{T}) where {T <: Real}
    (p, k, noise_ratio) = (inst.p, inst.k, inst.noise_ratio)
    @assert 0 < k <= p

    # sample components that will carry the signal
    signal_idxs = Distributions.sample(1:p, k, replace = false)
    if noise_ratio <= 0
        # noiseless model
        x = zeros(T, p)
        x[signal_idxs] = rand(T, k)
        sigma = x * x'
        sigma ./= tr(sigma)
    else
        # simulate some observations with noise
        x = randn(T, p, 100)
        sigma = x * x'
        y = rand(Distributions.Normal(zero(T), noise_ratio), k)
        sigma[signal_idxs, signal_idxs] .+= y * y'
        sigma ./= 100
    end

    dimx = Cones.svec_length(p)
    # x is the svec (lower triangle, row-wise) of the matrix solution
    c = Cones.smat_to_svec!(zeros(T, dimx), -sigma, sqrt(T(2)))
    b = T[1]
    A = zeros(T, 1, dimx)
    for i in 1:p
        A[sum(1:i)] = 1
    end
    hpsd = zeros(T, dimx)
    cones = Cones.Cone{T}[Cones.PosSemidefTri{T, T}(dimx)]

    if inst.use_epinorminfdual
        # l1 cone
        # double off-diagonals, which are already scaled by rt2
        Gl1vec = fill(-one(T), dimx)
        Cones.scale_svec!(Gl1vec, sqrt(T(2)))
        G = [
            sparse(-one(T) * I, dimx, dimx);
            spzeros(T, 1, dimx);
            Diagonal(Gl1vec);
            ]
        h = vcat(hpsd, T(k), zeros(T, dimx))
        push!(cones, Cones.EpiNormInf{T, T}(1 + dimx, use_dual = true))
    else
        l1 = Cones.scale_svec!(ones(T, dimx), sqrt(T(2)))
        G = [
            -I    spzeros(T, dimx, 2 * dimx);
            spzeros(T, 2 * dimx, dimx)    -I;
            spzeros(T, 1, dimx)    repeat(l1', 1, 2);
            ]
        A = [
            sparse(A)    spzeros(T, 1, 2 * dimx);
            -I    -I    I;
            ]
        c = vcat(c, zeros(T, 2 * dimx))
        b = vcat(b, zeros(T, dimx))
        h = vcat(hpsd, zeros(T, 2 * dimx), k)
        push!(cones, Cones.Nonnegative{T}(2 * dimx + 1))
    end

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end

function test_extra(
    inst::SparsePCANative{T},
    solve_stats::NamedTuple,
    ::NamedTuple,
    ) where T
    @test solve_stats.status == Solvers.Optimal
    if solve_stats.status == Solvers.Optimal && iszero(inst.noise_ratio)
        # check objective value is correct
        tol = eps(T)^0.25
        @test solve_stats.primal_obj ≈ -1 atol=tol rtol=tol
    end
end
