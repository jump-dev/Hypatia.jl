#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

loc = joinpath(pwd(), "data")

n = 1
d = 5
l = 6
u = 11

pts = vec(readcsv(joinpath(loc, "pts.txt"), Float64))
w = vec(readcsv(joinpath(loc, "w.txt"), Float64))
P = readcsv(joinpath(loc, "P.txt"), Float64)
P0 = readcsv(joinpath(loc, "P0.txt"), Float64)
pts[abs.(pts) .< 1e-10] = 0.0
w[abs.(w) .< 1e-10] = 0.0
P[abs.(P) .< 1e-10] = 0.0
P0[abs.(P0) .< 1e-10] = 0.0

numPolys = 2
degPolys = 5

LWts = [5,]
nu = numPolys*(l + sum(LWts))
gh_bnu = nu + 1.0
wtVals = 1 - pts.^2

PWts = [qr(diagm(sqrt.(wtVals[:,j]))*P[:,1:LWts[j]])[1] for j in 1:n]

A = repeat(speye(u), outer=(1, numPolys))
b = w
# c = genRandPolyVals(n, numPolys, degPolys, P0)
c = vec(readcsv(joinpath(loc, "c.txt"), Float64))
c[abs.(c) .< 1e-10] = 0.0


function eval_gh_SOSWt(txp, Pp)
    Lp = chol(Symmetric(Pp'*diagm(txp)*Pp))'
    Vp = Lp\Pp'
    VtVp = Vp'*Vp

    inpolyp = true
    gp = -diag(VtVp)
    Hp = VtVp.^2

    return (inpolyp, gp, Hp)
end


function eval_gh(tx)
    incone = true
    g = zeros(numPolys*u)
    H = zeros(numPolys*u, numPolys*u)
    L = zeros(numPolys*u, numPolys*u)

    off = 0
    for polyId in 1:numPolys
        txp = tx[off+1:off+u]
        (inp, gp, Hp) = eval_gh_SOSWt(txp, P)

        if inp
            for j in 1:n
                (inpWt, gpWt, HpWt) = eval_gh_SOSWt(txp, PWts[j])
                inp &= inpWt
                if inp
                    gp = gp + gpWt
                    Hp = Hp + HpWt
                else
                    error("failure in eval_gh")
                end
            end
        end

        if inp
            Lp = chol(Symmetric(Hp))'
            g[off+1:off+u] = gp
            H[off+1:off+u,off+1:off+u] = Hp
            L[off+1:off+u,off+1:off+u] = Lp
            off += u
        else
            error("failure in eval_gh")
        end
    end

    @assert issymmetric(H)

    return (incone, g, H, L)
end


# test
# tx0 = ones(numPolys*u)
# (incone0, g0, H0, L0) = eval_gh(tx0)
# @show incone0
# @show g0
# @show H0
# @show L0


# load into optimizer and solve
using Alfonso
using MathOptInterface



opt = AlfonsoOptimizer()
Alfonso.loaddata!(opt::AlfonsoOptimizer, A::AbstractMatrix, b::Vector{Float64}, c::Vector{Float64}, eval_gh::Function, gh_bnu::Float64)
@show opt
MathOptInterface.optimize!(opt)
