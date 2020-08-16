#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see "A Direct Formulation for Sparse PCA Using Semidefinite Programming" by
Alexandre d’Aspremont, Laurent El Ghaoui, Michael I. Jordan, Gert R. G. Lanckriet
=#

import Hypatia.BlockMatrix
import Distributions

struct SparsePCANative{T <: Real} <: ExampleInstanceNative{T}
    p::Int
    k::Int
    use_epinorminfdual::Bool # use dual of epinorminf cone, else nonnegative cones
    noise_ratio::Real
    use_linops::Bool
end

function build(inst::SparsePCANative{T}) where {T <: Real}
    (p, k, noise_ratio) = (inst.p, inst.k, inst.noise_ratio)
    @assert 0 < k <= p

    signal_idxs = Distributions.sample(1:p, k, replace = false) # sample components that will carry the signal
    if noise_ratio <= 0
        # noiseless model
        x = zeros(T, p)
        x[signal_idxs] = rand(T, k)
        sigma = x * x'
        sigma ./= tr(sigma)
    else
        # simulate some observations with noise
        x = randn(p, 100)
        sigma = x * x'
        y = rand(Distributions.Normal(0, Float64(noise_ratio)), k)
        sigma[signal_idxs, signal_idxs] .+= y * y'
        sigma ./= 100
        sigma = T.(sigma)
    end

    dimx = Cones.svec_length(p)
    # x will be the svec (lower triangle, row-wise) of the matrix solution we seek
    c = Cones.smat_to_svec!(zeros(T, dimx), -sigma, sqrt(T(2)))
    b = T[1]
    A = zeros(T, 1, dimx)
    for i in 1:p
        s = sum(1:i)
        A[s] = 1
    end
    hpsd = zeros(T, dimx)
    cones = Cones.Cone{T}[Cones.PosSemidefTri{T, T}(dimx)]

    if inst.use_epinorminfdual
        # l1 cone
        # double off-diagonals, which are already scaled by rt2
        if inst.use_linops
            Gl1 = Diagonal(-one(T) * I, dimx)
        else
            Gl1 = -Matrix{T}(I, dimx, dimx)
        end
        ModelUtilities.vec_to_svec!(Gl1, rt2 = sqrt(T(2)))
        if inst.use_linops
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
        push!(cones, Cones.EpiNormInf{T, T}(1 + dimx, use_dual = true))
    else
        id = Matrix{T}(I, dimx, dimx)
        l1 = ModelUtilities.vec_to_svec!(ones(T, dimx), rt2 = sqrt(T(2)))
        if inst.use_linops
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
        push!(cones, Cones.Nonnegative{T}(2 * dimx + 1))
    end

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end

function test_extra(inst::SparsePCANative, result::NamedTuple)
    @test result.status == :Optimal
    if result.status == :Optimal && iszero(inst.noise_ratio)
        # check objective value is correct
        tol = eps(eltype(result.x))^0.25
        @test result.primal_obj ≈ -1 atol = tol rtol = tol
    end
end

insts[SparsePCANative]["minimal"] = [
    ((3, 2, true, 0, false),),
    ((3, 2, false, 0, false),),
    ((3, 2, true, 10, false),),
    ((3, 2, false, 10, false),),
    ]
insts[SparsePCANative]["fast"] = [
    ((5, 3, true, 0, false),),
    ((5, 3, false, 0, false),),
    ((5, 3, true, 10, false),),
    ((5, 3, false, 10, false),),
    ((30, 10, true, 0, false),),
    ((30, 10, false, 0, false),),
    ((30, 10, true, 10, false),),
    ((30, 10, false, 10, false),),
    ]
insts[SparsePCANative]["slow"] = Tuple[]
insts[SparsePCANative]["linops"] = [
    ((5, 3, true, 0, true),),
    ((5, 3, false, 0, true),),
    ((5, 3, true, 10, true),),
    ((5, 3, false, 10, true),),
    ((30, 10, true, 0, true),),
    ((30, 10, false, 0, true),),
    ((30, 10, true, 10, true),),
    ((30, 10, false, 10, true),),
    ]
