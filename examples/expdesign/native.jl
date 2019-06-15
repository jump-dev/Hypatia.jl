#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see description in examples/expdesign/JuMP.jl

=#

using LinearAlgebra
import Random
using Test
import Hypatia
import Hypatia
const HYP = Hypatia
const MO = HYP.Models
const MU = HYP.ModelUtilities
const CO = HYP.Cones
const SO = HYP.Solvers

const rt2 = sqrt(2)

function expdesign(
    q::Int,
    p::Int,
    n::Int,
    nmax::Int;
    use_logdet::Bool = false,
    )

    @assert (p > q) && (n > q) && (nmax <= n)
    V = randn(q, p)

    # hypograph variable and number of trials of each experiment
    A = ones(1, p)
    b = Float64[n]

    # nonnegativity
    G_nonneg = -Matrix{Float64}(I, p, p)
    h_nonneg = zeros(p)
    # do <= nmax experiments
    G_nmax = Matrix{Float64}(I, p, p)
    h_nmax = fill(nmax, p)

    cones = CO.Cone[CO.Nonnegative{Float64}(p), CO.Nonnegative{Float64}(p)]
    cone_idxs = [1:p, (p + 1):(2 * p)]

    if use_logdet

        # pad with hypograph variable
        A = [0 A]
        G_nonneg = [zeros(p) G_nonneg]
        G_nmax = [zeros(p) G_nmax]
        # maximize the hypograph variable of the logdet cone
        c = [-1, zeros(p)...]

        # dimension of vectorized matrix V*diag(np)*V'
        dimvec = Int(q * (q + 1) / 2)
        G_logdet = zeros(dimvec, p)
        l = 1
        for i in 1:q, j in 1:i
            for k in 1:p
                G_logdet[l, k] = -V[i, k] * V[j, k] * (i == j ? 1 : rt2)
            end
            l += 1
        end
        @assert l - 1 == dimvec
        # pad with hypograph variable and perspective variable
        h_logdet = [0, 1, zeros(size(G_logdet, 1))...]
        G_logdet = [-1 zeros(1, p); zeros(1, p + 1); zeros(dimvec) G_logdet]
        push!(cones, CO.HypoPerLogdet{Float64}(dimvec + 2))
        push!(cone_idxs, (2 * p + 1):(2 * p + dimvec + 2))

        G = vcat(G_nonneg, G_nmax, G_logdet)
        h = vcat(h_nonneg, h_nmax, h_logdet)
    else

        # requires an upper triangular matrix of additional variables, ordered as vec(triangle)
        num_trivars = Int(q * (q + 1) / 2)
        # pad with triangle matrix variables and q hypopoerlog cone hypograph variables
        A = [A zeros(1, num_trivars) zeros(1, q)]
        G_nonneg = [G_nonneg zeros(p, num_trivars) zeros(p, q)]
        G_nmax = [G_nmax zeros(p, num_trivars) zeros(p, q)]
        # maximize the sum of hypograph variables of all hypoperlog cones
        c = vcat(zeros(p), zeros(num_trivars), -ones(q))

        # number of experiments, upper triangular matrix, hypograph variables
        dimx = p + num_trivars + q

        # vectorized dimension of psd matrix
        dimvec = q * (2 * q + 1)
        G_psd = zeros(dimvec, dimx)

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

        h_psd = zeros(dimvec)
        push!(cone_idxs, (2 * p + 1):(2 * p + dimvec))
        push!(cones, CO.PosSemidef{Float64, Float64}(dimvec))

        G_log = zeros(3 * q, dimx)
        h_log = zeros(3 * q)
        offset = 1
        for i in 1:q
            # hypograph variable
            G_log[offset, p + num_trivars + i] = -1
            # perspective variable
            h_log[offset + 1] = 1
            # diagonal element in the triangular matrix
            G_log[offset + 2, p + diag_idx(i)] = -1
            cone_offset = 2 * p + dimvec + offset
            push!(cone_idxs, cone_offset:(cone_offset + 2))
            push!(cones, CO.HypoPerLog{Float64}())
            offset += 3
        end

        G = vcat(G_nonneg, G_nmax, G_psd, G_log)
        h = vcat(h_nonneg, h_nmax, h_psd, h_log)

    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

function test_expdesign(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    model = MO.PreprocessedLinearModel{Float64}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{Float64}(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    return
end

expdesign1() = expdesign(25, 75, 125, 5, use_logdet = true)
expdesign2() = expdesign(10, 30, 50, 5, use_logdet = true)
expdesign3() = expdesign(5, 15, 25, 5, use_logdet = true)
expdesign4() = expdesign(4, 8, 12, 3, use_logdet = true)
expdesign5() = expdesign(3, 5, 7, 2, use_logdet = true)
expdesign6() = expdesign(25, 75, 125, 5, use_logdet = false)
expdesign7() = expdesign(10, 30, 50, 5, use_logdet = false)
expdesign8() = expdesign(5, 15, 25, 5, use_logdet = false)
expdesign9() = expdesign(4, 8, 12, 3, use_logdet = false)
expdesign10() = expdesign(3, 5, 7, 2, use_logdet = false)

test_expdesign_all(; options...) = test_expdesign.([
    expdesign1,
    expdesign2,
    expdesign3,
    expdesign4,
    expdesign5,
    expdesign6,
    expdesign7,
    expdesign9,
    expdesign9,
    expdesign10,
    ], options = options)

test_expdesign(; options...) = test_expdesign.([
    expdesign5,
    expdesign10,
    expdesign3,
    expdesign8,
    expdesign1,
    expdesign6,
    ], options = options)
