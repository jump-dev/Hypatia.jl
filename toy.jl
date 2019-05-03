using Random
using Distributions
using Combinatorics
import Hypatia
const MU = Hypatia.ModelUtilities
const MO = Hypatia.Models
const CO = Hypatia.Cones
const SO = Hypatia.Solvers
using LinearAlgebra
using DynamicPolynomials
Random.seed!(1234)

include("examples/polymin/jump.jl")
include("examples/polymin/real.jl")


# alfonso included
# polyname = :motzkin # [-1,1]^n
# polyname = :robinson # [-1,1]^n
# polyname = :rosenbrock # [-1,1]^n
# polyname = :schwefel # [-10,10]^n
# polyname = :caprasse # [-0.5, 0.5]^n
# polyname = :lotkavolterra
# polyname = :butcher
# polyname = :heart
# polyname = :magnetism7
# polyname = :reactiondiffusion

# for (polyname, (n, lbs, ubs, deg, fn)) in polys
#     solveandcheck(polyname)
# end

function solveandcheck(polyname)
    (n, lbs, ubs, deg, fn) = polys[polyname]
    (_, dppoly, _, truemin) = getpolydata(polyname)
    d = div(deg + 1, 2)
    L = binomial(n + d, n)
    U = binomial(n + 2d, n)
    g(x, n) = (x[n] - lbs[n]) * (ubs[n] - x[n])
    nwts = n + 1 # boxes only

    npts = U * 200
    pts0 = rand(Uniform(-1, 1), npts, n)

    # npts = prod((2d + 1):(2d + n))
    # pts0 = Matrix{Float64}(undef, npts, n)
    # for j in 1:n
    #     ig = prod((2d + 1 + j):(2d + n))
    #     cs = MU.cheb2_pts(2d + j)
    #     i = 1
    #     l = 1
    #     while true
    #         pts0[i:(i + ig - 1), j] .= cs[l]
    #         i += ig
    #         l += 1
    #         if l >= 2d + 1 + j
    #             if i >= npts
    #                 break
    #             end
    #             l = 1
    #         end
    #     end
    # end

    M = Matrix{Float64}(undef, npts, U)
    (keep_pts, _) = MU.choose_interp_pts!(M, pts0, 2d, U, false)
    pts_kept = pts0[keep_pts, :]
    P0 = M[keep_pts, 1:L]

    ptsi = zeros(size(pts_kept))
    ptsi .= pts_kept
    PWts = Vector{Matrix{Float64}}(undef, nwts - 1)
    for ni in 1:(nwts - 1)
        ptsi[:, ni] = pts_kept[:, ni] * (ubs[ni] - lbs[ni]) / 2 .+ (ubs[ni] + lbs[ni]) / 2
        g1vec = [sqrt(g(ptsi[i, :], ni)) for i in 1:U]
        L1 = binomial(n + d - 1, n)
        PWts[ni] = Diagonal(g1vec) * P0[:, 1:L1]
    end

    c = [-1.0]
    A = zeros(0, 1)
    b = Float64[]
    G = ones(U, 1)
    h = [fn(ptsi[j, :]...) for j in 1:U]

    cones = [CO.WSOSPolyInterp(U, [P0, PWts...], false)]
    cone_idxs = [1:U]
    model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = true)
    SO.solve(solver)
    s2 = solver.point.s
    z2 = solver.point.z
    @assert SO.get_status(solver) == :Optimal
    @assert isapprox(solver.point.x[1], truemin, atol = 1e-2) # TODO checkout numerically bad goldsteinprice, rosenbrock under BK





    lagrange_polys = MU.recover_lagrange_polys(ptsi, 2d)
    @show dot(lagrange_polys, solver.point.s) + Monomial(1) * solver.point.x[1] # yep motzkin
    @polyvar x[1:n]

    ds = vcat(d, ones(Int, n) * (d - 1))
    cheb_polys = [MU.get_chebyshev_polys(x, d) for d in ds]
    # cheb of inverse transform
    for cps in cheb_polys, p in eachindex(cps), i in 1:n
        if x[i] in variables(cps[p])
            cps[p] = subs(cps[p], x[i] => (x[i] - (ubs[i] + lbs[i]) / 2) ./ ((ubs[i] - lbs[i]) / 2))
        end
    end

    get_lambda(pt, P) = P' * Diagonal(pt) * P
    ipwt = [P0, PWts...]
    sprimal = solver.point.s
    sdual = solver.point.z
    cone = cones[1]
    cone.point .= sdual
    CO.check_in_cone(cone)
    H = Symmetric(cones[1].H, :U)
    w = H \ sprimal
    gram_matrices = Vector{Matrix{Float64}}(undef, nwts)
    for p in 1:nwts
        lambda_inv = inv(Symmetric(get_lambda(sdual, ipwt[p]), :U))
        lambdaw = get_lambda(w, ipwt[p]) #+ 1e-6I
        S = Symmetric(lambda_inv * lambdaw * lambda_inv, :U)
        gram_matrices[p] = S
    end
    box_weights = [g(x, ni) for ni in 1:n]
    weight_funs = [1; box_weights...]
    recovered_poly = sum(cheb_polys[p]' * gram_matrices[p] * cheb_polys[p] * weight_funs[p] for p in 1:nwts) + Monomial(1) * solver.point.x[1]
    recovered_coeffs = coefficients(recovered_poly)[abs.(coefficients(recovered_poly)) .> 1e-2] # TODO checkot numerically inaccurate e.g. heart, :schwefel under BK
    @show recovered_coeffs
    @show coefficients(dppoly)
    @assert isapprox(recovered_coeffs, coefficients(dppoly), atol = 1e-1)

    return
end


for polyname in keys(polys)
    @show polyname
    polyname == :reactiondiffusion && continue # degree 0 cheb polys
    solveandcheck(polyname)
end



;
