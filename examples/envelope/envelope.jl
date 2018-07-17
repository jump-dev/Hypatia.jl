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
L = 6
U = 11

pts = readcsv(joinpath(loc, "pts.txt"), Float64)'
w = readcsv(joinpath(loc, "w.txt"), Float64)'
P = readcsv(joinpath(loc, "P.txt"), Float64)
P0 = readcsv(joinpath(loc, "P0.txt"), Float64)
pts[abs.(pts) .< 1e-10] = 0.0
w[abs.(w) .< 1e-10] = 0.0
P[abs.(P) .< 1e-10] = 0.0
P0[abs.(P0) .< 1e-10] = 0.0

numPolys = 2
degPolys = 5

LWts = [5,]
nu = numPolys*(L + sum(LWts))
gh_bnu = nu + 1
wtVals = 1 - pts.^2

PWts = [qr(diagm(sqrt.(wtVals[:,j]))*P[:,1:LWts[j]])[1] for j in 1:n]

A = repeat(speye(U), outer=(1, numPolys))
b = w
# c = genRandPolyVals(n, numPolys, degPolys, P0)
c = readcsv(joinpath(loc, "c.txt"), Float64)
c[abs.(c) .< 1e-10] = 0.0


function eval_gh(tx)
    function eval_gh_SOSWt(txPoly, PPoly)
        Y = Symmetric(PPoly'*diagm(txPoly)*PPoly)
        L = chol(Y)'

        inpoly = true
        V = L\PPoly'
        VtV = V'*V

        g = -diag(VtV)
        H = VtV.^2

        # [L, err] = chol(Y, 'lower');
        # if err > 0
        #     in = 0;
        #     g = NaN;
        #     H = NaN;
        # else
        #     in = 1;
        #     V = L\P';
        #     VtV = V'*V;
        #     g = -diag(VtV);
        #     H = VtV.^2;
        # end

        return (inpoly, g, H)
    end

    incone = true
    g = zeros(numPolys*U)
    H = zeros(numPolys*U, numPolys*U)
    L = zeros(numPolys*U, numPolys*U)

    off = 0
    for polyId in 1:numPolys
        txPoly = tx[off+1:off+U]
        (inPoly, gPoly, HPoly) = eval_gh_SOSWt(txPoly, P)

        if inPoly
            for j in 1:n
                # for the weight 1-t_j^2
                (inPolyWt, gPolyWt, HPolyWt) = eval_gh_SOSWt(txPoly, PWts[j])
                inPoly = inPoly && inPolyWt
                if inPoly
                    gPoly = gPoly + gPolyWt
                    HPoly = HPoly + HPolyWt
                else
                    error("failure in eval_gh")
                    # gPoly = NaN
                    # HPoly = NaN
                    # break
                end
            end
        end

        if inPoly
            LPoly = chol(HPoly)' # TODO symmetric?
            @show LPoly
            @show L
            g[off+1:off+U] = gPoly
            H[off+1:off+U,off+1:off+U] = HPoly
            L[off+1:off+U,off+1:off+U] = LPoly
            off += U
        else
            error("failure in eval_gh")
            # incone = false
            break
        end
    end

    return (incone, g, H, L)
end

# test
tx0 = ones(numPolys*U)
(incone0, g0, H0, L0) = eval_gh(tx0)
@show incone0
@show g0
@show H0
@show L0
