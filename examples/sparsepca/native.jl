#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see "A Direct Formulation for Sparse PCA Using Semidefinite Programming" by
Alexandre d’Aspremont, Laurent El Ghaoui, Michael I. Jordan, Gert R. G. Lanckriet
==#

using LinearAlgebra
import Random
using Test
import Hypatia
const HYP = Hypatia
const MO = HYP.Models
const MU = HYP.ModelUtilities
const CO = HYP.Cones
const SO = HYP.Solvers

const rt2 = sqrt(2)

# function sparsepca(n::Int)
#     A = randn(n, n)
#     A = A + A'
#     model = JuMP.Model()
#     JuMP.@variable(model, X[i in 1:n, j in 1:i])
#     @show vcat(1, [X[i, j] * (i == j ? 1 : 2) for i in 1:n for j in 1:i])
#     JuMP.@constraints(model, begin
#         vcat(k, [X[i, j] * (i == j ? 1 : 2) for i in 1:n for j in 1:i]) in CO.EpiNormInf{Float64}(1 + div(n * (n + 1), 2), true)
#         [X[i, j] * (i == j ? 1 : rt2) for i in 1:n for j in 1:i] in MOI.PositiveSemidefiniteConeTriangle(n)
#         sum(X[i, i] for i in 1:n) == 1
#     end)
#     JuMP.@objective(model, Max, sum(X[i, j] * A[i, j] * (i == j ? 1 : 2) for i in 1:n for j in 1:i))
#     return (model = model,)
# end
function sparsepca(n::Int, k::Int; T = Float64, mat = zeros(T, 0, 0))
    dimx = div(n * (n + 1), 2)
    if isempty(mat)
        mat = randn(n, n)
        mat = mat + mat'
    else
        @assert size(mat) == (n, n)
    end
    c = [mat[i, j] for i in 1:n for j in 1:i]
    b = T[1]
    A = zeros(T, 1, dimx)
    # l1 cone
    G1 = -Matrix{T}(I, dimx, dimx) * 2
    # PSD cone
    G2 = -Matrix{T}(I, dimx, dimx) * rt2
    for i in 1:n
        s = sum(1:i)
        A[s] = 1
        G1[s, s] = -1
        G2[s, s] = -1
    end
    G = vcat(zeros(T, 1, dimx), G1, G2)
    h = vcat(k, zeros(2 * dimx))
    cones = [CO.EpiNormInf{T}(1 + dimx, true), CO.PosSemidef{T, T}(dimx)]
    cone_idxs = [1:(dimx + 1), (dimx + 2):(2 * dimx + 1)]
    @show (A), (G), cone_idxs, size(h), size(c)
    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

sparsepca1(; T::THR = Float) = sparsepca(3, 3, T = T)
# sparsepca2(; T::THR = Float) = sparsepca(3, 3, T = T, mat = Matrix{Float64}(I, 3, 3))

function test_sparsepca(instance::Function; T::THR = Float64, rseed::Int = 1, options)
    Random.seed!(rseed)
    d = instance(T = T)
    model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{T}(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    @show r.x
    return
end

test_sparsepca_all(; options...) = test_sparsepca.([
    sparsepca1,
    ], options = options)

test_sparsepca1(; options...) = test_sparsepca.([
    sparsepca1,
    ], options = options)
