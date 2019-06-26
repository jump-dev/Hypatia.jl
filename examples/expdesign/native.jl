#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see description in examples/expdesign/JuMP.jl
=#

using LinearAlgebra
import Random
using Test
import Hypatia
import Hypatia.HypReal
const HYP = Hypatia
const MO = HYP.Models
const CO = HYP.Cones
const SO = HYP.Solvers

function expdesign(
    T::Type{<:HypReal},
    q::Int,
    p::Int,
    n::Int,
    nmax::Int;
    use_logdet::Bool = true,
    use_sumlog::Bool = true,
    )
    @assert (p > q) && (n > q) && (nmax <= n)
    rt2 = sqrt(T(2))
    V = T(4) * rand(T, q, p) .- T(2)

    # hypograph variable and number of trials of each experiment
    A = ones(T, 1, p)
    b = T[n]

    # nonnegativity
    G_nonneg = Matrix{T}(-I, p, p)
    h_nonneg = zeros(T, p)
    # do <= nmax experiments
    G_nmax = Matrix{T}(-I, p, p)
    h_nmax = fill(T(nmax), p)

    cones = CO.Cone{T}[CO.Nonnegative{T}(p), CO.Nonnegative{T}(p)]
    cone_idxs = [1:p, (p + 1):(2 * p)]

    if use_logdet
        # pad with hypograph variable
        A = hcat(zero(T), A)
        G_nonneg = hcat(zeros(T, p), G_nonneg)
        G_nmax = hcat(zeros(T, p), G_nmax)
        # maximize the hypograph variable of the logdet cone
        c = vcat(-one(T), zeros(T, p))

        # dimension of vectorized matrix V*diag(np)*V'
        dimvec = div(q * (q + 1), 2)
        G_logdet = zeros(T, dimvec, p)
        l = 1
        for i in 1:q, j in 1:i
            for k in 1:p
                G_logdet[l, k] = -V[i, k] * V[j, k] * (i == j ? 1 : rt2)
            end
            l += 1
        end
        @assert l - 1 == dimvec
        # pad with hypograph variable and perspective variable
        h_logdet = vcat(zero(T), one(T), zeros(T, dimvec))
        G_logdet = [
            -one(T)    zeros(T, 1, p);
            zeros(T, 1, p + 1);
            zeros(T, dimvec)    G_logdet;
            ]
        push!(cones, CO.HypoPerLogdet{T}(dimvec + 2))
        push!(cone_idxs, (2 * p + 1):(2 * p + dimvec + 2))

        G = vcat(G_nonneg, G_nmax, G_logdet)
        h = vcat(h_nonneg, h_nmax, h_logdet)
    else
        # requires an upper triangular matrix of additional variables, ordered row wise
        num_trivars = div(q * (q + 1), 2)
        if use_sumlog
            dimx = p + num_trivars + 1
            padx = num_trivars + 1
            num_hypo = 1
        else
            # number of experiments, upper triangular matrix, hypograph variables
            dimx = p + num_trivars + q
            padx = num_trivars + q
            num_hypo = q
        end
        # pad with triangle matrix variables and q hypopoerlog cone hypograph variables
        A = [A zeros(T, 1, padx)]
        G_nonneg = hcat(G_nonneg, zeros(T, p, padx))
        G_nmax = hcat(G_nmax, zeros(T, p, padx))
        # maximize the sum of hypograph variables of all hypoperlog cones
        c = vcat(zeros(T, p + num_trivars), -ones(T, num_hypo))

        # vectorized dimension of psd matrix
        dimvec = q * (2 * q + 1)
        G_psd = zeros(T, dimvec, dimx)

        # variables in upper triangular matrix numbered row-wise
        diag_idx(i::Int) = (i == 1 ? 1 : 1 + sum(q - j for j in 0:(i - 2)))

        # V*diag(np)*V
        l = 1
        for i in 1:q, j in 1:i
            for k in 1:p
                G_psd[l, k] = -V[i, k] * V[j, k] * (i == j ? 1 : rt2)
            end
            l += 1
        end
        # [triangle' diag(triangle)]
        tri_idx = 1
        for i in 1:q
            # triangle'
            # skip zero-valued elements
            l += i - 1
            for j in i:q
                G_psd[l, p + tri_idx] = -rt2
                l += 1
                tri_idx += 1
            end
            # diag(triangle)
            # skip zero-valued elements
            l += i - 1
            G_psd[l, p + diag_idx(i)] = -1
            l += 1
        end

        h_psd = zeros(T, dimvec)
        push!(cones, CO.PosSemidef{T, T}(dimvec))
        push!(cone_idxs, (2 * p + 1):(2 * p + dimvec))

        if use_sumlog
            G_logvars = zeros(T, q, num_trivars)
            for i in 1:q
                G_logvars[i, diag_idx(i)] = -1
            end
            G_log = [
                # hypograph row
                zeros(T, 1, p + num_trivars)    -one(T)
                # perspective row
                zeros(T, 1, dimx)
                # log row
                zeros(T, q, p)    G_logvars    zeros(T, q)
                ]
            h_log = vcat(zero(T), one(T), zeros(T, q))
            push!(cones, CO.HypoPerSumLog{T}(q + 2))
            push!(cone_idxs, (2 * p + dimvec + 1):(2 * p + dimvec + 2 + q))
        else
            G_log = zeros(T, 3 * q, dimx)
            h_log = zeros(T, 3 * q)
            offset = 1
            for i in 1:q
                # hypograph variable
                G_log[offset, p + num_trivars + i] = -1
                # perspective variable
                h_log[offset + 1] = 1
                # diagonal element in the triangular matrix
                G_log[offset + 2, p + diag_idx(i)] = -1
                cone_offset = 2 * p + dimvec + offset
                push!(cones, CO.HypoPerLog{T}())
                push!(cone_idxs, cone_offset:(cone_offset + 2))
                offset += 3
            end
        end
        G = vcat(G_nonneg, G_nmax, G_psd, G_log)
        h = vcat(h_nonneg, h_nmax, h_psd, h_log)
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

function test_expdesign(T::Type{<:HypReal}, instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{T}(model; options...)
    SO.solve(solver)
    tol = max(1e-5, sqrt(sqrt(eps(T))))
    r = SO.get_certificates(solver, model, test = true, atol = tol, rtol = tol)
    @test r.status == :Optimal
    return
end

expdesign1(T::Type{<:HypReal}) = expdesign(T, 25, 75, 125, 5, use_logdet = true)
expdesign2(T::Type{<:HypReal}) = expdesign(T, 10, 30, 50, 5, use_logdet = true)
expdesign3(T::Type{<:HypReal}) = expdesign(T, 5, 15, 25, 5, use_logdet = true)
expdesign4(T::Type{<:HypReal}) = expdesign(T, 4, 8, 12, 3, use_logdet = true)
expdesign5(T::Type{<:HypReal}) = expdesign(T, 3, 5, 7, 2, use_logdet = true)
expdesign6(T::Type{<:HypReal}) = expdesign(T, 25, 75, 125, 5, use_logdet = false)
expdesign7(T::Type{<:HypReal}) = expdesign(T, 10, 30, 50, 5, use_logdet = false)
expdesign8(T::Type{<:HypReal}) = expdesign(T, 5, 15, 25, 5, use_logdet = false)
expdesign9(T::Type{<:HypReal}) = expdesign(T, 4, 8, 12, 3, use_logdet = false)
expdesign10(T::Type{<:HypReal}) = expdesign(T, 3, 5, 7, 2, use_logdet = false)

test_expdesign_all(T::Type{<:HypReal}; options...) = test_expdesign.(T, [
    expdesign1,
    expdesign2,
    expdesign3,
    expdesign4,
    expdesign5,
    expdesign6,
    expdesign7,
    expdesign8,
    expdesign9,
    expdesign10,
    ], options = options)

test_expdesign(T::Type{<:HypReal}; options...) = test_expdesign.(T, [
    expdesign5,
    expdesign10,
    ], options = options)
