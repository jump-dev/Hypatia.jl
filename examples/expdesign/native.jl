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
    use_logdet::Bool = true,
    )

    @assert (p > q) && (n > q) && (nmax <= n)
    V = randn(q, p)

    # hypograph variable and number of trials of each experiment
    A = [0 ones(p)...]
    b = Float64[n]
    c = [-1, zeros(p)...]

    # nonnegativity
    G_nonneg = [zeros(p) -Matrix{Float64}(I, p, p)]
    h_nonneg = zeros(p)
    # do <= nmax experiments
    G_nmax = [zeros(p) Matrix{Float64}(I, p, p)]
    h_nmax = fill(nmax, p)

    cones = CO.Cone[CO.Nonnegative{Float64}(p), CO.Nonnegative{Float64}(p)]
    cone_idxs = [1:p, (p + 1):(2 * p)]

    if use_logdet
        # dimension of vectorized matrix V*diag(np)*V'
        dimvec = Int(q * (q + 1) / 2)
        G_logdet = zeros(dimvec, p)
        l = 1
        for i in 1:q, j in 1:i
            G_logdet[l, :] = -[V[i, k] * V[j, k] for k in 1:p] * (i == j ? 1 : rt2)
            l += 1
        end
        @assert l - 1 == dimvec
        # pad with hypograph variable and perspective variable
        h_logdet = [0, 1, zeros(size(G_logdet, 1))...]
        G_logdet = [-1 zeros(1, p); zeros(1, p + 1); zeros(dimvec) G_logdet]
        push!(cones, CO.HypoPerLogdet{Float64}(dimvec + 2))
        push!(cone_idxs, (2 * p + 1):(2 * p + dimvec + 2))
    else
        # something
    end

    G = vcat(G_nonneg, G_nmax, G_logdet)
    h = vcat(h_nonneg, h_nmax, h_logdet)

    @show size(G), size(A), cone_idxs, size(h), size(c)

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

expdesign1() = expdesign(25, 75, 125, 5) # large
expdesign2() = expdesign(10, 30, 50, 5) # medium
expdesign3() = expdesign(5, 15, 25, 5) # small
expdesign4() = expdesign(4, 8, 12, 3) # tiny
expdesign5() = expdesign(3, 5, 7, 2) # miniscule
expdesign6() = expdesign(25, 75, 125, 5, use_logdet = false) # large
expdesign7() = expdesign(10, 30, 50, 5, use_logdet = false) # medium
expdesign8() = expdesign(5, 15, 25, 5, use_logdet = false) # small
expdesign9() = expdesign(4, 8, 12, 3, use_logdet = false) # tiny
expdesign10() = expdesign(3, 5, 7, 2, use_logdet = false) # miniscule

test_expdesign_all(; options...) = test_expdesign.([
    expdesign1,
    expdesign2,
    expdesign3,
    expdesign4,
    expdesign5,
    # expdesign6,
    # expdesign7,
    # expdesign9,
    # expdesign9,
    # expdesign10,
    ], options = options)

test_expdesign(; options...) = test_expdesign.([
    expdesign3,
    # expdesign8,
    ], options = options)
