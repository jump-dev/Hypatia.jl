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
function sparsepca(n::Int, k::Int; T = Float64, mat = zeros(T, 0, 0), use_l1ball::Bool = true)
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
    # PSD cone
    Gpsd = -Matrix{T}(I, dimx, dimx) * rt2
    for i in 1:n
        s = sum(1:i)
        A[s] = 1
        Gpsd[s, s] = -1
    end
    hpsd = zeros(dimx)
    cones = CO.Cone[CO.PosSemidef{T, T}(dimx)]
    cone_idxs = [1:dimx]

    if use_l1ball
        # l1 cone
        Gl1 = -Matrix{T}(I, dimx, dimx) * 2
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
        A_l1 = [zeros(1, dimx) ones(1, 2 * dimx)]
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

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

sparsepca1(; T = Float) = sparsepca(3, 3, T = T)
sparsepca2(; T = Float) = sparsepca(3, 3, T = T, use_l1ball = false)
# sparsepca2(; T::THR = Float) = sparsepca(3, 3, T = T, mat = Matrix{Float64}(I, 3, 3))

function test_sparsepca(instance::Function; T = Float64, rseed::Int = 1, options)
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
    sparsepca2,
    ], options = options)

test_sparsepca(; options...) = test_sparsepca.([
    sparsepca1,
    sparsepca2,
    ], options = options)
