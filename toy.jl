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

motzkin = ((x,y) -> 1-48x^2*y^2+64x^2*y^4+64x^4*y^2)

# box on domain [-10,10], [-0.1, 0.1]

n = 2
lbs = [-4; -1]
ubs = [4; 1]
deg = 6
d = 3
L = binomial(n + d, n)
U = binomial(n + 2d, n)
npts = 200
g1(x) = (x[1] - lbs[1]) * (ubs[1] - x[1])
g2(x) = (x[2] - lbs[2]) * (ubs[2] - x[2])


# first set of points
pts0 = rand(Uniform(-1, 1), npts, 2)
M = Matrix{Float64}(undef, npts, U)
(keep_pts, _) = MU.choose_interp_pts!(M, pts0, deg, U, false)
pts_kept = pts0[keep_pts, :]
P0 = M[keep_pts, 1:L]

# try either this
pts1 = zeros(size(pts_kept))
pts1 .= pts_kept
pts1[:, 1] = pts_kept[:, 1] * (ubs[1] - lbs[1]) / 2
g1vec = [sqrt(g1(pts1[i, :])) for i in 1:U]
# or this
# g1vec = [sqrt(g1(pts_kept[i, :])) for i in 1:U]
#
L1 = binomial(n + d - 1, n)
P1 = Diagonal(g1vec) * M[keep_pts, 1:L1]

# either this
pts2 = zeros(size(pts_kept))
pts2 .= pts_kept
pts2[:, 2] = pts_kept[:, 2] * (ubs[2] - lbs[2]) / 2
g2vec = [sqrt(g2(pts2[i, :])) for i in 1:U]
L2 = binomial(n + d - 1, n)
P2 = Diagonal(g2vec) * M[keep_pts, 1:L2]

c = [-1.0]
A = zeros(0, 1)
b = Float64[]
G = ones(U, 1)
h = [motzkin(pts_kept[j, :]...) for j in 1:U]

cones = [CO.WSOSPolyInterp(U, [P0, P1, P2], false)]
cone_idxs = [1:U]
model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
solver = SO.HSDSolver(model, verbose = true)
SO.solve(solver)
s2 = solver.point.s
z2 = solver.point.z
# @test SO.get_status(solver) == :Optimal





lagrange_polys = MU.recover_lagrange_polys(pts_kept, deg)
dot(lagrange_polys, solver.point.s) # yep motzkin
@polyvar x[1:n]

ds = [3; 2; 2]
cheb_polys = [MU.get_chebyshev_polys(x, d) for d in ds]

get_lambda(pt, P) = P' * Diagonal(pt) * P
ipwt = [P0, P1, P2]
sprimal = solver.point.s
sdual = solver.point.z
cone = cones[1]
cone.point .= sdual
CO.check_in_cone(cone)
H = Symmetric(cones[1].H, :U)
nwts = 3
w = H \ sprimal
gram_matrices = Vector{Matrix{Float64}}(undef, nwts)
for p in 1:nwts
    lambda_inv = inv(Symmetric(get_lambda(sdual, ipwt[p]), :U))
    lambdaw = get_lambda(w, ipwt[p]) #+ 1e-6I
    S = Symmetric(lambda_inv * lambdaw * lambda_inv, :U)
    gram_matrices[p] = S
end
weight_funs = [
    1;
    (ubs[1] - x[1] * (ubs[1] - lbs[1]) / 2) * (x[1] * (ubs[1] - lbs[1]) / 2 - lbs[1]);
    (ubs[2] - x[2]) * (x[2] - lbs[2])
    ]
# weight_funs = [1; (ubs[1] - x[1]) * (x[1] - lbs[1]); (ubs[2] - x[2]) * (x[2] - lbs[2])]
@show sum(cheb_polys[p]' * gram_matrices[p] * cheb_polys[p] * weight_funs[p] for p in 1:nwts)

;
