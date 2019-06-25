#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

polyminreal: formulates and solves the real polynomial optimization problem for a given polynomial; see:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming.

polymincomplex: minimizes a real-valued complex polynomial over a domain defined by real-valued complex polynomials

TODO
- generalize ModelUtilities interpolation code for complex polynomials space
- merge real and complex polyvars data when complex is supported in DynamicPolynomials: https://github.com/JuliaAlgebra/MultivariatePolynomials.jl/issues/11
=#

import Random
using LinearAlgebra
import Combinatorics
using Test
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models
const SO = HYP.Solvers
const MU = HYP.ModelUtilities

const rt2 = sqrt(2)
# TODO remove when converting cones is implemented
T = Float64

function polyminreal(
    n::Int,
    halfdeg::Int;
    use_primal::Bool = false,
    use_wsos::Bool = true,
    )
    if use_primal && !use_wsos
        error("primal psd formulation is not implemented yet")
    end

    dom = MU.Box(-ones(n), ones(n))
    (U, pts, P0, PWts, _) = MU.interpolate(dom, halfdeg, sample = true)
    interp_vals = randn(U)

    if use_wsos
        cones = [CO.WSOSPolyInterp{T, T}(U, [P0, PWts...], !use_primal)]
        cone_idxs = [1:U]
    else
        # will be set up iteratively
        cones = CO.Cone[]
        cone_idxs = UnitRange{Int}[]
    end

    if use_primal
        c = [-1.0]
        A = zeros(0, 1)
        b = Float64[]
        G = ones(U, 1)
        h = interp_vals
        true_obj *= -1
    else
        c = interp_vals
        A = ones(1, U) # TODO eliminate constraint and first variable
        b = [1.0]
        if use_wsos
            G = Diagonal(-1.0I, U) # TODO use UniformScaling
            h = zeros(U)
        else
            G = zeros(0, U)
            rowidx = 1
            for Pk in [P0, PWts...]
                Lk = size(Pk, 2)
                dk = Int(Lk * (Lk + 1) / 2)
                push!(cone_idxs, rowidx:(rowidx + dk - 1))
                push!(cones, CO.PosSemidef{Float64, Float64}(dk))
                Gk = Matrix{Float64}(undef, dk, U)
                l = 1
                for i in 1:Lk, j in 1:i
                    for u in 1:U
                        Gk[l, u] = -Pk[u, i] * Pk[u, j] * (i == j ? 1 : rt2)
                    end
                    l += 1
                end
                G = vcat(G, Gk)
                rowidx += dk
            end
            h = zeros(size(G, 1))
        end
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end


# n_range = [2, 3, 4]
# halfdeg_range = [3, 4, 5]
# T = Float64
#
# io = open("polyminreal.csv", "w")
# println(io, "usewsos,seed,n,halfdeg,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
# for n in n_range, halfdeg in halfdeg_range, use_wsos in tf, seed in seeds
#
#     Random.seed!(seed)
#     d = polyminreal(n, halfdeg, use_wsos = use_wsos)
#     model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
#     solver = SO.HSDSolver{T}(model, tol_abs_opt = 1e-5, tol_rel_opt = 1e-5, time_limit = 600)
#     t = @timed SO.solve(solver)
#     r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
#     dimx = size(d.G, 2)
#     dimy = size(d.A, 1)
#     dimz = size(d.G, 1)
#     println(io, "$use_wsos,$seed,$n,$halfdeg,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
#         "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
#         "$(solver.y_feas),$(solver.z_feas)"
#         )
#
# end
# close(io)
