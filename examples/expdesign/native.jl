#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see description in examples/expdesign/JuMP.jl
=#

using LinearAlgebra
import Random
using Test
import Hypatia
import Hypatia
const HYP = Hypatia
const MO = HYP.Models
const MU = HYP.ModelUtilities
const CO = HYP.Cones
const SO = HYP.Solvers

const rt2 = sqrt(2)

function expdesign(
    q::Int,
    p::Int,
    n::Int,
    nmax::Int;
    use_logdet::Bool = true,
    T::DataType = Float64,
    )
    @assert (p > q) && (n > q) && (nmax <= n)
    V = randn(q, p)

    # hypograph variable and number of trials of each experiment
    A = ones(T, 1, p)
    b = T[n]

    # nonnegativity
    G_nonneg = -Matrix{T}(I, p, p)
    h_nonneg = zeros(T, p)
    # do <= nmax experiments
    G_nmax = Matrix{T}(I, p, p)
    h_nmax = fill(T(nmax), p)

    cones = CO.Cone[CO.Nonnegative{T}(p), CO.Nonnegative{T}(p)]
    cone_idxs = [1:p, (p + 1):(2 * p)]

    if use_logdet
        # pad with hypograph variable
        A = T[0 A]
        G_nonneg = [zeros(T, p) G_nonneg]
        G_nmax = [zeros(T, p) G_nmax]
        # maximize the hypograph variable of the logdet cone
        c = vcat(T(-1), zeros(T, p))

        # dimension of vectorized matrix V*diag(np)*V'
        dimvec = div(q * (q + 1), 2)
        G_logdet = zeros(T, dimvec, p)
        l = 1
        for i in 1:q, j in 1:i
            for k in 1:p
                G_logdet[l, k] = -V[i, k] * V[j, k] * (i == j ? 1 : rt2)
            end
            l += 1
        end
        @assert l - 1 == dimvec
        # pad with hypograph variable and perspective variable
        h_logdet = [T(0), T(1), zeros(T, size(G_logdet, 1))...]
        G_logdet = [T(-1) zeros(T, 1, p); zeros(T, 1, p + 1); zeros(T, dimvec) G_logdet]
        push!(cones, CO.HypoPerLogdet{T}(dimvec + 2))
        push!(cone_idxs, (2 * p + 1):(2 * p + dimvec + 2))

        G = vcat(G_nonneg, G_nmax, G_logdet)
        h = vcat(h_nonneg, h_nmax, h_logdet)
    else
        # requires an upper triangular matrix of additional variables, ordered row wise
        num_trivars = div(q * (q + 1), 2)
        # pad with triangle matrix variables and q hypopoerlog cone hypograph variables
        A = [A zeros(T, 1, num_trivars) zeros(T, 1, q)]
        G_nonneg = [G_nonneg zeros(T, p, num_trivars) zeros(T, p, q)]
        G_nmax = [G_nmax zeros(T, p, num_trivars) zeros(T, p, q)]
        # maximize the sum of hypograph variables of all hypoperlog cones
        c = vcat(zeros(T, p), zeros(T, num_trivars), -ones(T, q))

        # number of experiments, upper triangular matrix, hypograph variables
        dimx = p + num_trivars + q

        # vectorized dimension of psd matrix
        dimvec = q * (2 * q + 1)
        G_psd = zeros(T, dimvec, dimx)

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

        h_psd = zeros(T, dimvec)
        push!(cone_idxs, (2 * p + 1):(2 * p + dimvec))
        push!(cones, CO.PosSemidef{T, T}(dimvec))

        G_log = zeros(T, 3 * q, dimx)
        h_log = zeros(T, 3 * q)
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

        G = vcat(G_nonneg, G_nmax, G_psd, G_log)
        h = vcat(h_nonneg, h_nmax, h_psd, h_log)
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

function test_expdesign(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    T = Float64
    d = instance()
    model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver{T}(model; options...)
    @time SO.solve(solver)
    @show solver.num_iters
    r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    return
end

expdesign1() = expdesign(25, 75, 125, 5, use_logdet = true)
expdesign2() = expdesign(10, 30, 50, 5, use_logdet = true)
expdesign3() = expdesign(5, 15, 25, 5, use_logdet = true)
expdesign4() = expdesign(4, 8, 12, 3, use_logdet = true)
expdesign5() = expdesign(3, 5, 7, 2, use_logdet = true)
expdesign6() = expdesign(25, 75, 125, 5, use_logdet = false)
expdesign7() = expdesign(10, 30, 50, 5, use_logdet = false)
expdesign8() = expdesign(5, 15, 25, 5, use_logdet = false)
expdesign9() = expdesign(4, 8, 12, 3, use_logdet = false)
expdesign10() = expdesign(3, 5, 7, 2, use_logdet = false)

expdesign11() = expdesign(50, 100, 125, 5, use_logdet = true)
expdesign12() = expdesign(50, 100, 125, 5, use_logdet = false)

test_expdesign_all(; real_type::DataType = Float64, options...) = test_expdesign.([
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
    ], real_type = real_type, options = options)

test_expdesign(; options...) = test_expdesign.([
    expdesign3,
    # expdesign6,
    # expdesign3,
    # expdesign8,
    # expdesign4,
    # expdesign9,
    # expdesign5,
    # expdesign10,
    ], options = options)

@testset "" begin
    test_expdesign(verbose = true, tol_abs_opt = 1e-5, tol_rel_opt = 1e-5)
end




# n_range = [4, 8, 12]
# p_range = [4, 8, 16]

# q_range = [4, 6, 8]
# nmax = 5
# tf = [true]
# seeds = 1:2
# real_types = [Float64]
#
# io = open("expdesign.csv", "w")
# println(io, "uselogdet,real,seed,q,p,n,dimx,dimy,dimz,time,bytes,numiters,status,pobj,dobj,xfeas,yfeas,zfeas")
# for q in q_range, T in real_types, seed in seeds
#     p = 2 * q
#     n = 2 * q
#     Random.seed!(seed)
#     for use_logdet in tf
#         d = expdesign(q, p, n, nmax, use_logdet = use_logdet, T = T)
#         model = MO.PreprocessedLinearModel{T}(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
#         solver = SO.HSDSolver{T}(model, tol_abs_opt = 1e-5, tol_rel_opt = 1e-5)
#         t = @timed SO.solve(solver)
#         r = SO.get_certificates(solver, model, test = false, atol = 1e-4, rtol = 1e-4)
#         dimx = size(d.G, 2)
#         dimy = size(d.A, 1)
#         dimz = size(d.G, 1)
#         println(io, "$use_logdet,$T,$seed,$q,$p,$n,$dimx,$dimy,$dimz,$(t[2]),$(t[3])," *
#             "$(solver.num_iters),$(r.status),$(r.primal_obj),$(r.dual_obj),$(solver.x_feas)," *
#             "$(solver.y_feas),$(solver.z_feas)"
#             )
#     end
# end
# close(io)
#
#
# ;
