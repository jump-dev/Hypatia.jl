#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

given a sequence of observations X_1,...,X_n with each Xᵢ in Rᵈ,
find a density function f maximizing the log likelihood of the observations
    min -∑ᵢ zᵢ
    -zᵢ + log(f(Xᵢ)) ≥ 0 ∀ i = 1,...,n
    ∫f = 1
    f ≥ 0
==#

using LinearAlgebra
import Random
using Test
import DataFrames
import CSV
import DynamicPolynomials
import Hypatia
const HYP = Hypatia
const MO = HYP.Models
const MU = HYP.ModelUtilities
const CO = HYP.Cones
const SO = HYP.Solvers

function densityest(
    X::AbstractMatrix{Float64},
    deg::Int;
    sample_factor::Int = 100,
    use_sumlog::Bool = true,
    )
    (nobs, dim) = size(X)

    domain = MU.Box(-ones(dim), ones(dim))
    # rescale X to be in unit box
    minX = minimum(X, dims = 1)
    maxX = maximum(X, dims = 1)
    X .-= (minX + maxX) / 2
    X ./= (maxX - minX) / 2

    halfdeg = div(deg + 1, 2)
    (U, pts, P0, PWts, w) = MU.interpolate(domain, halfdeg, sample = true, calc_w = true, sample_factor = sample_factor)
    lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)
    basis_evals = Matrix{Float64}(undef, nobs, U)
    for i in 1:nobs, j in 1:U
        basis_evals[i, j] = lagrange_polys[j](X[i, :])
    end

    if use_sumlog
        c = vcat([-1.0], zeros(U))
        dimx = 1 + U
        A = zeros(1, dimx)
        A[1, 2:end] = w
        b = [1.0]
        h = zeros(U + 2 + nobs)
        G1 = zeros(U, dimx)
        G1[:, 2:end] = -Matrix{Float64}(I, U, U)
        G2 = zeros(2 + nobs, dimx)
        G2[1, 1] = -1.0
        h[2] = 1.0
        for i in 1:nobs
            G2[i + 2, 2:end] = -basis_evals[i, :]
        end
        G = vcat(G1, G2)
        cone_idxs = [1:U, (U + 1):(U + 2 + nobs)]
        @show cone_idxs, size(G1), size(G2), size(h), size(c), size(A), size(b)
        cones = [CO.WSOSPolyInterp{Float64, Float64}(U, [P0, PWts...]), CO.HypoPerSumLog{Float64}(nobs + 2)]
    else
        c = vcat(-ones(nobs), zeros(U))
        dimx = nobs + U
        A = zeros(1, dimx)
        A[1, (nobs + 1):end] = w
        b = [1.0]
        h = zeros(U + 3 * nobs)
        G1 = zeros(U, dimx)
        G1[:, (nobs + 1):end] = -Matrix{Float64}(I, U, U)
        G2 = zeros(3 * nobs, dimx)
        offset = 1
        for i in 1:nobs
            G2[offset, i] = -1.0
            h[U + offset + 1] = 1.0
            G2[offset + 2, (nobs + 1):end] = -basis_evals[i, :]
            offset += 3
        end
        G = vcat(G1, G2)
        cone_idxs = vcat([1:U], [(3 * (i - 1) + U + 1):(3 * i + U) for i in 1:nobs])
        # @show dimx, size(A)
        cones = vcat(CO.WSOSPolyInterp{Float64, Float64}(U, [P0, PWts...]), [CO.HypoPerLog{Float64}() for _ in 1:nobs])
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

densityest(nobs::Int, n::Int, deg::Int) = densityest(randn(nobs, n), deg)

# iris dataset
function iris_data()
    df = CSV.read(joinpath(@__DIR__, "data", "iris.csv"))
    # only use setosa species
    dfsub = df[df.species .== "setosa", [:sepal_length, :sepal_width, :petal_length, :petal_width]] # n = 4
    X = convert(Matrix{Float64}, dfsub)
    return X
end

# lung cancer dataset from https://github.com/therneau/survival (cancer.rda)
# description at https://github.com/therneau/survival/blob/master/man/lung.Rd
function cancer_data()
    df = CSV.read(joinpath(@__DIR__, "data", "cancer.csv"), missingstring = "NA", copycols = true)
    DataFrames.dropmissing!(df, disallowmissing = true)
    # only use males with status 2
    dfsub = df[df.status .== 2, :]
    dfsub = dfsub[dfsub.sex .== 1, [:time, :age, :ph_ecog, :ph_karno, :pat_karno, :meal_cal, :wt_loss]] # n = 7
    X = convert(Matrix{Float64}, dfsub)
    return X
end

densityest1() = densityest(iris_data(), 4)
densityest2() = densityest(iris_data(), 6)
densityest3() = densityest(200, 1, 4)

function test_densityest(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    model = MO.PreprocessedLinearModel{Float64}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{Float64}(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    return
end

test_densityest_all(; options...) = test_densityest.([
    densityest1,
    densityest2,
    densityest3,
    ], options = options)

test_densityest(; options...) = test_densityest.([
    # densityest1,
    densityest3,
    ], options = options)
