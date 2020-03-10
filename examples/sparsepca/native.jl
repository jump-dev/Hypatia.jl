#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see "A Direct Formulation for Sparse PCA Using Semidefinite Programming" by
Alexandre d’Aspremont, Laurent El Ghaoui, Michael I. Jordan, Gert R. G. Lanckriet
==#

include(joinpath(@__DIR__, "../common_native.jl"))
import Hypatia.BlockMatrix
import Distributions

function sparsepca_native(
    ::Type{T},
    p::Int,
    k::Int,
    use_epinorminfdual::Bool, # use dual of epinorminf cone, else nonnegative cones
    noise_ratio::Real,
    use_linops::Bool,
    ) where {T <: Real}
    @assert 0 < k <= p

    signal_idxs = Distributions.sample(1:p, k, replace = false) # sample components that will carry the signal
    if noise_ratio <= 0
        # noiseless model
        x = zeros(T, p)
        x[signal_idxs] = rand(T, k)
        sigma = x * x'
        sigma ./= tr(sigma)
        true_obj = -1
    else
        # simulate some observations with noise
        x = randn(p, 100)
        sigma = x * x'
        y = rand(Distributions.Normal(0, Float64(noise_ratio)), k)
        sigma[signal_idxs, signal_idxs] .+= y * y'
        sigma ./= 100
        sigma = T.(sigma)
        true_obj = NaN
    end

    dimx = CO.svec_length(p)
    # x will be the svec (lower triangle, row-wise) of the matrix solution we seek
    c = CO.smat_to_svec!(zeros(T, dimx), -sigma, sqrt(T(2)))
    b = T[1]
    A = zeros(T, 1, dimx)
    for i in 1:p
        s = sum(1:i)
        A[s] = 1
    end
    hpsd = zeros(T, dimx)
    cones = CO.Cone{T}[CO.PosSemidefTri{T, T}(dimx)]

    if use_epinorminfdual
        # l1 cone
        # double off-diagonals, which are already scaled by rt2
        if use_linops
            Gl1 = Diagonal(-one(T) * I, dimx)
        else
            Gl1 = -Matrix{T}(I, dimx, dimx)
        end
        MU.vec_to_svec!(Gl1, rt2 = sqrt(T(2)))
        if use_linops
            G = BlockMatrix{T}(
                2 * dimx + 1,
                dimx,
                [-I, Gl1],
                [1:dimx, (dimx + 2):(2 * dimx + 1)],
                [1:dimx, 1:dimx]
                )
            A = BlockMatrix{T}(1, dimx, [A], [1:1], [1:dimx])
        else
            G = [
                Matrix{T}(-I, dimx, dimx); # psd cone
                zeros(T, 1, dimx);
                Gl1;
                ]
        end
        h = vcat(hpsd, T(k), zeros(T, dimx))
        push!(cones, CO.EpiNormInf{T, T}(1 + dimx, use_dual = true))
    else
        id = Matrix{T}(I, dimx, dimx)
        l1 = MU.vec_to_svec!(ones(T, dimx), rt2 = sqrt(T(2)))
        if use_linops
            A = BlockMatrix{T}(
                dimx + 1,
                3 * dimx,
                [A, -I, -I, I],
                [1:1, 2:(dimx + 1), 2:(dimx + 1), 2:(dimx + 1)],
                [1:dimx, 1:dimx, (dimx + 1):(2 * dimx), (2 * dimx + 1):(3 * dimx)]
                )
            G = BlockMatrix{T}(
                3 * dimx + 1,
                3 * dimx,
                [-I, -I, repeat(l1', 1, 2)],
                [1:dimx, (dimx + 1):(3 * dimx), (3 * dimx + 1):(3 * dimx + 1)],
                [1:dimx, (dimx + 1):(3 * dimx), (dimx + 1):(3 * dimx)]
                )
        else
            A = T[
                A    zeros(T, 1, 2 * dimx);
                -id    -id    id;
                ]
            G = [
                Matrix{T}(-I, dimx, dimx)    zeros(T, dimx, 2 * dimx);
                zeros(T, 2 * dimx, dimx)    Matrix{T}(-I, 2 * dimx, 2 * dimx);
                zeros(T, 1, dimx)    repeat(l1', 1, 2);
                ]
        end
        c = vcat(c, zeros(T, 2 * dimx))
        b = vcat(b, zeros(T, dimx))
        h = vcat(hpsd, zeros(T, 2 * dimx), k)
        push!(cones, CO.Nonnegative{T}(2 * dimx + 1))
    end

    model = Models.Model{T}(c, A, b, G, h, cones)
    return (model, (true_obj = true_obj,))
end

function test_sparsepca_native(result, test_helpers, test_options)
    @test result.status == :Optimal
    if result.status == :Optimal && !isnan(test_helpers.true_obj)
        # check objective value is correct
        tol = eps(eltype(result.x))^0.2
        @test result.primal_obj ≈ test_helpers.true_obj atol = tol rtol = tol
    end
end

options = ()
sparsepca_native_fast = [
    ((Float64, 5, 3, true, 0, false), (), options),
    ((Float64, 5, 3, false, 0, false), (), options),
    ((Float64, 5, 3, true, 10, false), (), options),
    ((Float64, 5, 3, false, 10, false), (), options),
    ((Float64, 30, 10, true, 0, false), (), options),
    ((Float64, 30, 10, false, 0, false), (), options),
    ((Float64, 30, 10, true, 10, false), (), options),
    ((Float64, 30, 10, false, 10, false), (), options),
    ]
sparsepca_native_slow = [
    # TODO
    ]
sparsepca_native_linops = [
    ((Float64, 5, 3, true, 0, true), (), options),
    ((Float64, 5, 3, false, 0, true), (), options),
    ((Float64, 5, 3, true, 10, true), (), options),
    ((Float64, 5, 3, false, 10, true), (), options),
    ((Float64, 30, 10, true, 0, true), (), options),
    ((Float64, 30, 10, false, 0, true), (), options),
    ((Float64, 30, 10, true, 10, true), (), options),
    ((Float64, 30, 10, false, 10, true), (), options),
    ]

# @testset begin "sparsepca_native" test_native_instance.(sparsepca_native, test_sparsepca_native, sparsepca_native_fast) end
;
