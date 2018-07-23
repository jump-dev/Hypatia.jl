#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Alfonso
import MathOptInterface
const MOI = MathOptInterface
using Printf
using SparseArrays
using LinearAlgebra
using DelimitedFiles

loc = joinpath(pwd(), "data")

n = 1
d = 5
l = 6
u = 11

pts = vec(readdlm(joinpath(loc, "pts.txt"), ',', Float64))
w = vec(readdlm(joinpath(loc, "w.txt"), ',', Float64))
P = readdlm(joinpath(loc, "P.txt"), ',', Float64)
P0 = readdlm(joinpath(loc, "P0.txt"), ',', Float64)
pts[abs.(pts) .< 1e-10] .= 0.0
w[abs.(w) .< 1e-10] .= 0.0
P[abs.(P) .< 1e-10] .= 0.0
P0[abs.(P0) .< 1e-10] .= 0.0

numPolys = 2
degPolys = 5

LWts = [5,]
gh_bnu = numPolys*(l + sum(LWts)) + 1.0
wtVals = 1.0 .- pts.^2

PWts = [Array((qr(Diagonal(sqrt.(wtVals[:,j]))*P[:,1:LWts[j]])).Q) for j in 1:n]

A = repeat(sparse(1.0I, u, u), outer=(1, numPolys))
b = w
# c = genRandPolyVals(n, numPolys, degPolys, P0)
c = vec(readdlm(joinpath(loc, "c.txt"), ',', Float64))
c[abs.(c) .< 1e-10] .= 0.0


function eval_gh(g, Hi, L, tx)
    off = 0
    for polyId in 1:numPolys
        txp = view(tx, off+1:off+u)
        gp = view(g, off+1:off+u)
        Hip = view(Hi, off+1:off+u, off+1:off+u)
        Lp = view(L, off+1:off+u, off+1:off+u)

        F = cholesky!(Symmetric(P'*Diagonal(txp)*P), check=false) # TODO could this cholesky of P'DP be faster?
        if !issuccess(F)
            return false
        end
        Vp = F.L\P'
        VtVp = Vp'*Vp
        gp .= -diag(VtVp)
        Lp .= VtVp.^2

        for j in 1:n
            F = cholesky!(Symmetric(PWts[j]'*Diagonal(txp)*PWts[j]), check=false)
            if !issuccess(F)
                return false
            end
            Vp = F.L\PWts[j]'
            VtVp = Vp'*Vp
            gp .-= diag(VtVp)
            Lp .+= VtVp.^2
        end

        F = cholesky!(Lp, check=false)
        if !issuccess(F)
            return false
        end
        Hip .= inv(F)
        Lp .= F.L

        off += u
    end
    return true
end





# load into optimizer and solve
opt = AlfonsoOptimizer(maxiter=100)
Alfonso.loaddata!(opt::AlfonsoOptimizer, A::AbstractMatrix{Float64}, b::Vector{Float64}, c::Vector{Float64}, eval_gh::Function, gh_bnu::Float64)
# @show opt
@time MOI.optimize!(opt)
