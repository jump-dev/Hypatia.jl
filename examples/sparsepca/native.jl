#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see "A Direct Formulation for Sparse PCA Using Semidefinite Programming" by
Alexandre d’Aspremont, Laurent El Ghaoui, Michael I. Jordan, Gert R. G. Lanckriet

TODO add examples with stochastic data e.g.
@assert 0 < k <= p
signal = randn(0, snr)
# sample components that will carry the signal
spike = sample(1:p, k)
sigma = zeros(p, p)
for i in 1:n
    x = randn(p)
    x[spike] .+= signal
    sigma += x * x' / n
end
==#

using LinearAlgebra
import Distributions
import Random
using Test
import Hypatia
const HYP = Hypatia
const MO = HYP.Models
const MU = HYP.ModelUtilities
const CO = HYP.Cones
const SO = HYP.Solvers

const rt2 = sqrt(2)

function sparsepca(mat::Matrix, k::Int; T = Float64, use_l1ball::Bool = true, spike = [])
    n = size(mat, 1)
    dimx = div(n * (n + 1), 2)
    # x will be the svec (lower triangle, row-wise) of the matrix solution we seek
    c = -[mat[i, j] * (i == j ? 1 : rt2) for i in 1:n for j in 1:i]
    b = T[1]
    A = zeros(T, 1, dimx)
    # PSD cone, x is already vectorized and scaled
    Gpsd = -Matrix{T}(I, dimx, dimx)
    for i in 1:n
        s = sum(1:i)
        A[s] = 1
    end
    hpsd = zeros(dimx)
    cones = CO.Cone[CO.PosSemidef{T, T}(dimx)]
    cone_idxs = [1:dimx]

    if use_l1ball
        # l1 cone
        Gl1 = -Matrix{T}(I, dimx, dimx) * rt2 # need to double off-diagonals, which are already scaled by rt2
        for i in 1:n
            s = sum(1:i)
            Gl1[s, s] = -1
        end
        G = vcat(Gpsd, zeros(T, 1, dimx), Gl1)
        h = vcat(hpsd, k, zeros(T, dimx))
        push!(cones, CO.EpiNormInf{T}(1 + dimx, true))
        push!(cone_idxs, (dimx + 1):(2 * dimx + 1))
    else
        c = vcat(c, zeros(2 * dimx))
        id = Matrix{T}(I, dimx, dimx)
        A_slacks = [-id -id id]
        A_l1 = [zeros(1, dimx) [i == j ? 1 : rt2]] # need to double off-diagonals, which are already scaled by rt2
        A = vcat([A zeros(T, 1, 2 * dimx)], A_slacks, A_l1)
        b = vcat(b, zeros(T, dimx), k)
        G = [
            Gpsd zeros(T, dimx, 2 * dimx)
            zeros(2 * dimx, dimx) -Matrix{T}(I, 2 * dimx, 2 * dimx)
            ]
        h = vcat(hpsd, zeros(2 * dimx))
        push!(cones, CO.Nonnegative{T}(2 * dimx))
        push!(cone_idxs, (dimx + 1):(3 * dimx))
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs, mat = mat, spike = spike)
end

function sparsepca(n::Int, p::Int, k::Int; snr = 100, T = Float64, use_l1ball::Bool = true)
    x = zeros(p)
    signal = Distributions.sample(1:p, k, replace = false)
    x[signal] = randn(k)
    sigma = x * x'
    sigma ./= tr(sigma)
    return sparsepca(sigma, k, T = Float64, use_l1ball = true, spike = x)
end

sparsepca1(; T = Float) = sparsepca(3, 3, 3, T = T)
sparsepca2(; T = Float) = sparsepca(3, 3, 3, T = T, use_l1ball = false)
sparsepca3(; T = Float) = sparsepca(10, 10, 3, T = T)
sparsepca4(; T = Float) = sparsepca(10, 10, 3, T = T, use_l1ball = false)

function test_sparsepca(instance::Function; T = Float64, rseed::Int = 1, options)
    Random.seed!(rseed)
    d = instance(T = T)
    model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{T}(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    return
end

test_sparsepca_all(; options...) = test_sparsepca.([
    sparsepca1,
    sparsepca2,
    sparsepca3,
    sparsepca4,
    ], options = options)

test_sparsepca(; options...) = test_sparsepca.([
    sparsepca1,
    sparsepca2,
    ], options = options)
