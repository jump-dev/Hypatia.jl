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

function polyminreal(
    n::Int,
    halfdeg::Int;
    use_primal::Bool = false,
    use_wsos::Bool = true,
    # T = Float64,
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


n_range = [2, 3, 4]
halfdeg_range = [3, 4, 5]
T = Float64

io = open("polyminreal.csv", "w")
println(io, "usewsos,seed,n,halfdeg,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
for n in n_range, halfdeg in halfdeg_range, use_wsos in tf, seed in seeds

    Random.seed!(seed)
    d = polyminreal(n, halfdeg, use_wsos = use_wsos)
    model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{T}(model, tol_abs_opt = 1e-5, tol_rel_opt = 1e-5, time_limit = 600)
    t = @timed SO.solve(solver)
    r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
    dimx = size(d.G, 2)
    dimy = size(d.A, 1)
    dimz = size(d.G, 1)
    println(io, "$use_wsos,$seed,$n,$halfdeg,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
        "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
        "$(solver.y_feas),$(solver.z_feas)"
        )

end
close(io)




# function polymincomplex(
#     polyname::Symbol,
#     halfdeg::Int;
#     use_primal::Bool = true,
#     use_wsos::Bool = true,
#     sample_factor::Int = 100,
#     use_QR::Bool = false,
#     )
#     if !use_wsos
#         error("psd formulation is not implemented yet")
#     end
#
#     (n, f, gs, g_halfdegs, true_obj) = complexpolys[polyname]
#
#     # generate interpolation
#     # TODO use more numerically-stable basis for columns
#     L = binomial(n + halfdeg, n)
#     U = L^2
#     L_basis = [a for t in 0:halfdeg for a in Combinatorics.multiexponents(n, t)]
#     mon_pow(z, ex) = prod(z[i]^ex[i] for i in eachindex(ex))
#     V_basis = [z -> mon_pow(z, L_basis[k]) * mon_pow(conj(z), L_basis[l]) for l in eachindex(L_basis) for k in eachindex(L_basis)]
#     @assert length(V_basis) == U
#
#     # sample from domain (inefficient for general domains, only samples from unit box and checks feasibility)
#     num_samples = sample_factor * U
#     samples = Vector{Vector{ComplexF64}}(undef, num_samples)
#     k = 0
#     randbox() = 2 * rand() - 1
#     while k < num_samples
#         z = [Complex(randbox(), randbox()) for i in 1:n]
#         if all(g -> g(z) > 0.0, gs)
#             k += 1
#             samples[k] = z
#         end
#     end
#
#     # select subset of points to maximize |det(V)| in heuristic QR-based procedure (analogous to real case)
#     V = [b(z) for z in samples, b in V_basis]
#     @test rank(V) == U
#     VF = qr(Matrix(transpose(V)), Val(true))
#     keep = VF.p[1:U]
#     points = samples[keep]
#     V = V[keep, :]
#     @test rank(V) == U
#
#     # setup P matrices
#     P0 = V[:, 1:L]
#     if use_QR
#         P0 = Matrix(qr(P0).Q)
#     end
#     P_data = [P0]
#     for i in eachindex(gs)
#         gi = gs[i].(points)
#         Pi = Diagonal(sqrt.(gi)) * P0[:, 1:binomial(n + halfdeg - g_halfdegs[i], n)]
#         if use_QR
#             Pi = Matrix(qr(Pi).Q)
#         end
#         push!(P_data, Pi)
#     end
#
#     # setup problem data
#     if use_primal
#         c = [-1.0]
#         A = zeros(0, 1)
#         b = Float64[]
#         G = ones(U, 1)
#         h = f.(points)
#         true_obj *= -1
#     else
#         c = f.(points)
#         A = ones(1, U) # TODO can eliminate equality and a variable
#         b = [1.0]
#         G = Diagonal(-1.0I, U)
#         h = zeros(U)
#     end
#     cones = [CO.WSOSPolyInterp{Float64, ComplexF64}(U, P_data, !use_primal)]
#     cone_idxs = [1:U]
#
#     return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs, true_obj = true_obj)
# end
#
# polymincomplex1() = polymincomplex(:abs1d, 1)
# polymincomplex2() = polymincomplex(:absunit1d, 1)
# polymincomplex3() = polymincomplex(:negabsunit1d, 2)
# polymincomplex4() = polymincomplex(:absball2d, 1)
# polymincomplex5() = polymincomplex(:absbox2d, 2)
# polymincomplex6() = polymincomplex(:negabsbox2d, 1)
# polymincomplex7() = polymincomplex(:denseunit1d, 2)
# polymincomplex8() = polymincomplex(:abs1d, 1, use_primal = false)
# polymincomplex9() = polymincomplex(:absunit1d, 1, use_primal = false)
# polymincomplex10() = polymincomplex(:negabsunit1d, 2, use_primal = false)
# polymincomplex11() = polymincomplex(:absball2d, 1, use_primal = false)
# polymincomplex12() = polymincomplex(:absbox2d, 2, use_primal = false)
# polymincomplex13() = polymincomplex(:negabsbox2d, 1, use_primal = false)
# polymincomplex14() = polymincomplex(:denseunit1d, 2, use_primal = false)
#
# function test_polymin(instance::Function; options, rseed::Int = 1)
#     Random.seed!(rseed)
#     d = instance()
#     model = MO.PreprocessedLinearModel{Float64}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
#     solver = SO.HSDSolver{Float64}(model; options...)
#     SO.solve(solver)
#     r = SO.get_certificates(solver, model, test = true, atol = 1e-4, rtol = 1e-4)
#     @test r.status == :Optimal
#     @test r.primal_obj ≈ d.true_obj atol = 1e-4 rtol = 1e-4
#     return
# end
#
