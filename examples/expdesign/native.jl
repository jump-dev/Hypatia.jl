#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see description in examples/expdesign/JuMP.jl
=#

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

THR = Type{<: HYP.HypReal}

function expdesign(
    q::Int,
    p::Int,
    n::Int,
    nmax::Int;
    T::THR = Float64,
    use_logdet::Bool = true,
    use_sumlog::Bool = true,
    alpha = 4,
    )
    @assert (p > q) && (n > q) && (nmax <= n)
    V = randn(q, p)

    # hypograph variable and number of trials of each experiment
    A = ones(1, p)
    b = Float64[n]

    # nonnegativity
    G_nonneg = -Matrix{Float64}(I, p, p)
    h_nonneg = zeros(p)
    # do <= nmax experiments
    G_nmax = -Matrix{Float64}(I, p, p)
    h_nmax = fill(nmax, p)

    cones = CO.Cone[CO.Nonnegative{T}(p), CO.Nonnegative{T}(p)]
    cone_idxs = [1:p, (p + 1):(2 * p)]

    if use_logdet
        # pad with hypograph variable
        A = [0 A]
        G_nonneg = [zeros(p) G_nonneg]
        G_nmax = [zeros(p) G_nmax]
        # maximize the hypograph variable of the logdet cone
        c = vcat(-1, zeros(p))

        # dimension of vectorized matrix V*diag(np)*V'
        dimvec = div(q * (q + 1), 2)
        G_logdet = zeros(dimvec, p)
        l = 1
        for i in 1:q, j in 1:i
            for k in 1:p
                G_logdet[l, k] = -V[i, k] * V[j, k] * (i == j ? 1 : rt2)
            end
            l += 1
        end
        @assert l - 1 == dimvec
        # pad with hypograph variable and perspective variable
        h_logdet = [0, 1, zeros(size(G_logdet, 1))...]
        G_logdet = [-1 zeros(1, p); zeros(1, p + 1); zeros(dimvec) G_logdet]
        push!(cones, CO.HypoPerLogdet{T}(dimvec + 2, alpha = alpha))
        push!(cone_idxs, (2 * p + 1):(2 * p + dimvec + 2))

        G = vcat(G_nonneg, G_nmax, G_logdet)
        h = vcat(h_nonneg, h_nmax, h_logdet)
    else
        # requires an upper triangular matrix of additional variables, ordered row wise
        num_trivars = div(q * (q + 1), 2)
        if use_sumlog
            dimx = p + num_trivars + 1
            padx = num_trivars + 1
            num_hypo = 1
        else
            # number of experiments, upper triangular matrix, hypograph variables
            dimx = p + num_trivars + q
            padx = num_trivars + q
            num_hypo = q
        end
        # pad with triangle matrix variables and q hypopoerlog cone hypograph variables
        A = [A zeros(1, padx)]
        G_nonneg = [G_nonneg zeros(p, padx)]
        G_nmax = [G_nmax zeros(p, padx)]
        # maximize the sum of hypograph variables of all hypoperlog cones
        c = vcat(zeros(p), zeros(num_trivars), -ones(num_hypo))

        # vectorized dimension of psd matrix
        dimvec = q * (2 * q + 1)
        G_psd = zeros(dimvec, dimx)

        # variables in upper triangular matrix numbered row-wise
        diag_idx(i::Int) = (i == 1 ? 1 : 1 + sum(q - j for j in 0:(i - 2)))

        # V*diag(np)*V
        l = 1
        for i in 1:q, j in 1:i
            for k in 1:p
                G_psd[l, k] = -V[i, k] * V[j, k] * (i == j ? 1 : rt2)
            end
            l += 1
        end
        # [triangle' diag(triangle)]
        tri_idx = 1
        for i in 1:q
            # triangle'
            # skip zero-valued elements
            l += i - 1
            for j in i:q
                G_psd[l, p + tri_idx] = -rt2
                l += 1
                tri_idx += 1
            end
            # diag(triangle)
            # skip zero-valued elements
            l += i - 1
            G_psd[l, p + diag_idx(i)] = -1
            l += 1
        end

        h_psd = zeros(dimvec)
        push!(cone_idxs, (2 * p + 1):(2 * p + dimvec))
        push!(cones, CO.PosSemidef{T, T}(dimvec))

        if use_sumlog
            G_logvars = zeros(q, num_trivars)
            for i in 1:q
                G_logvars[i, diag_idx(i)] = -1
            end
            G_log = [
                # hypograph row
                zeros(1, p + num_trivars) -1
                # perspective row
                zeros(1, dimx)
                # log row
                zeros(q, p) G_logvars zeros(q)
                ]
            h_log = vcat(0, 1, zeros(q))
            push!(cone_idxs, (2 * p + dimvec + 1):(2 * p + dimvec + 2 + q))
            push!(cones, CO.HypoPerSumLog{T}(q + 2, alpha = alpha))
        else
            G_log = zeros(3 * q, dimx)
            h_log = zeros(3 * q)
            offset = 1
            for i in 1:q
                # hypograph variable
                G_log[offset, p + num_trivars + i] = -1
                # perspective variable
                h_log[offset + 1] = 1
                # diagonal element in the triangular matrix
                G_log[offset + 2, p + diag_idx(i)] = -1
                cone_offset = 2 * p + dimvec + offset
                push!(cone_idxs, cone_offset:(cone_offset + 2))
                push!(cones, CO.HypoPerLog{T}())
                offset += 3
            end
        end
        G = vcat(G_nonneg, G_nmax, G_psd, G_log)
        h = vcat(h_nonneg, h_nmax, h_psd, h_log)
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

function test_expdesign(instance::Function; T::THR = Float64, rseed::Int = 1, options)
    Random.seed!(rseed)
    d = instance(T = T)
    model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{T}(model; options...)
    SO.solve(solver)
    r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    return
end

expdesign1(; T::THR = Float64) = expdesign(25, 75, 125, 5, use_logdet = true, T = T)
expdesign2(; T::THR = Float64) = expdesign(10, 30, 50, 5, use_logdet = true, T = T)
expdesign3(; T::THR = Float64) = expdesign(5, 15, 25, 5, use_logdet = true, T = T)
expdesign4(; T::THR = Float64) = expdesign(4, 8, 12, 3, use_logdet = true, T = T)
expdesign5(; T::THR = Float64) = expdesign(3, 5, 7, 2, use_logdet = true, T = T)
expdesign6(; T::THR = Float64) = expdesign(25, 75, 125, 5, use_logdet = false, T = T)
expdesign7(; T::THR = Float64) = expdesign(10, 30, 50, 5, use_logdet = false, T = T)
expdesign8(; T::THR = Float64) = expdesign(5, 15, 25, 5, use_logdet = false, T = T)
expdesign9(; T::THR = Float64) = expdesign(4, 8, 12, 3, use_logdet = false, T = T)
expdesign10(; T::THR = Float64) = expdesign(3, 5, 7, 2, use_logdet = false, T = T)

test_expdesign_all(; T::THR = Float64, options...) = test_expdesign.([
    expdesign1,
    expdesign2,
    expdesign3,
    expdesign4,
    expdesign5,
    expdesign6,
    expdesign7,
    expdesign9,
    expdesign9,
    expdesign10,
    ], T = T, options = options)

test_expdesign(; T::THR = Float64, options...) = test_expdesign.([
    expdesign3,
    expdesign8,
    ], T = T, options = options)




# q_range = [4]
# nmax = 5
# tf = [true, false]
# seeds = 1:2
# real_types = [BigFloat]
# alpha_range = [4]
#
# # compile run
# (p, q, n) = (3, 2, nmax)
# for T in real_types, use_logdet in [false], use_sumlog in tf
#     d = expdesign(q, p, n, nmax, use_logdet = use_logdet, use_sumlog = use_sumlog, T = T)
#     model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
#     solver = SO.HSDSolver{T}(model, tol_abs_opt = 1e-5, tol_rel_opt = 1e-5)
#     t = @timed SO.solve(solver)
#     r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
# end
#
# io = open("expdesign.csv", "w")
# println(io, "uselogdet,use_sumlog,real,alpha,seed,q,p,n,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
# for q in q_range, T in real_types, seed in seeds, use_logdet in [false], use_sumlog in tf
#     p = 2 * q
#     n = 2 * q
#
#     if use_logdet || use_sumlog
#         a_range = alpha_range
#     else
#         a_range = [-1]
#     end
#     for alpha in a_range
#         Random.seed!(seed)
#         d = expdesign(q, p, n, nmax, use_logdet = use_logdet, use_sumlog = use_sumlog, T = T, alpha = alpha)
#         model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
#         solver = SO.HSDSolver{T}(model, tol_abs_opt = 1e-5, tol_rel_opt = 1e-5, time_limit = 600)
#         t = @timed SO.solve(solver)
#         r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
#         dimx = size(d.G, 2)
#         dimy = size(d.A, 1)
#         dimz = size(d.G, 1)
#         println(io, "$use_logdet,$use_sumlog,$T,$alpha,$seed,$q,$p,$n,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
#             "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
#             "$(solver.y_feas),$(solver.z_feas)"
#             )
#     end
#
# end
# close(io)


;
