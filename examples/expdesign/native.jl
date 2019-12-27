#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see description in examples/expdesign/JuMP.jl
=#

using LinearAlgebra
import Random
using Test
import Hypatia
import Hypatia.BlockMatrix
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities
const HYP = Hypatia

function expdesign(
    T::Type{<:Real},
    q::Int,
    p::Int,
    n::Int,
    nmax::Int;
    use_linops::Bool = false,
    logdet_obj::Bool = false,
    rootdet_obj::Bool = false,
    geomean_obj::Bool = false,
    use_logdet::Bool = true,
    use_sumlog::Bool = true,
    use_rootdet::Bool = true,
    )
    @assert xor(logdet_obj, geomean_obj, rootdet_obj)
    @assert (p > q) && (n > q) && (nmax <= n)
    V = T(4) * rand(T, q, p) .- T(2)

    if use_linops && !logdet_obj
        error("linear operators only implemented with logdet objective")
    end

    A = ones(T, 1, p)
    b = T[n]
    h_nonneg = zeros(T, p)
    h_nmax = fill(T(nmax), p)
    cones = CO.Cone{T}[CO.Nonnegative{T}(p), CO.Nonnegative{T}(p)]

    if (logdet_obj && use_logdet) || (rootdet_obj && use_rootdet) && !use_linops
        # maximize the hypograph variable of the cone
        c = vcat(-one(T), zeros(T, p))

        # pad with hypograph variable
        A = hcat(zero(T), A)
        G_nonneg = hcat(zeros(T, p), Matrix{T}(-I, p, p))
        G_nmax = hcat(zeros(T, p), Matrix{T}(-I, p, p))

        # dimension of vectorized matrix V*diag(np)*V'
        dimvec = CO.svec_length(q)
        G_detcone = zeros(T, dimvec, p)
        l = 1
        for i in 1:q, j in 1:i
            for k in 1:p
                G_detcone[l, k] = -V[i, k] * V[j, k]
            end
            l += 1
        end
        MU.vec_to_svec!(G_detcone, rt2 = sqrt(T(2)))
        @assert l - 1 == dimvec

        if logdet_obj
            push!(cones, CO.HypoPerLogdetTri{T, T}(dimvec + 2))
            # pad with hypograph variable and perspective variable
            h_detcone = vcat(zero(T), one(T), zeros(T, dimvec))
            # include perspective variable
            G_detcone = [
                -one(T)    zeros(T, 1, p);
                zeros(T, 1, p + 1);
                zeros(T, dimvec)    G_detcone;
                ]
        else
            push!(cones, CO.HypoRootdetTri{T, T}(dimvec + 1))
            # pad with hypograph variable
            h_detcone = zeros(T, dimvec + 1)
            # include perspective variable
            G_detcone = [
                -one(T)    zeros(T, 1, p);
                zeros(T, dimvec)    G_detcone;
                ]
        end # logdet/rootdet
        # all conic constraints
        G = vcat(G_nonneg, G_nmax, G_detcone)
        h = vcat(h_nonneg, h_nmax, h_detcone)

        return (c = c, A = A, b = b, G = G, h = h, cones = cones, p = p)
    end

    if geomean_obj
        # auxiliary matrix variable has pq elements represented row-major, auxiliary lower triangular variable has svec_length(q) elements also row-major
        pq = p * q
        qq = q ^ 2
        num_trivars = CO.svec_length(q)
        c = vcat(-one(T), zeros(T, p + pq + num_trivars))

        A_VW = zeros(T, qq, pq)
        A_lowertri = zeros(T, qq, num_trivars)
        # rows index (i, j)
        row_idx = 1
        for i in 1:q
            col_offset = sum(1:(i - 1)) + 1
            A_lowertri[row_idx:(row_idx + i - 1), col_offset:(col_offset + i - 1)] = -Matrix{T}(-I, i, i)
            for j in 1:q
                for k in 1:p
                    # columns index (k, j)
                    col_idx = (k - 1) * q + j
                    A_VW[row_idx, col_idx] = V[i, k]
                end
                row_idx += 1
            end
        end

        A = [
            zero(T)    A    zeros(1, pq + num_trivars);
            zeros(T, qq, 1 + p)    A_VW    A_lowertri;
            ]
        b = vcat(b, zeros(T, qq))

        push!(cones, CO.HypoGeomean{T}(fill(inv(T(p)), p)))
        G_soc_epi = zeros(T, pq + p, p)
        G_soc = zeros(T, pq + p, pq)
        epi_idx = 1
        col_idx = 1
        for i in 1:p
            push!(cones, CO.EpiNormEucl{T}(q + 1))
            G_soc_epi[epi_idx, i] = -sqrt(T(q))
            G_soc[(epi_idx + 1):(epi_idx + q), col_idx:(col_idx + q - 1)] = Matrix{T}(-I, q, q)
            epi_idx += q + 1
            col_idx += q
        end
        G = [
            zeros(T, p)    Matrix{T}(-I, p, p)    zeros(T, p, pq + num_trivars); # nonnegativity
            zeros(T, p)    Matrix{T}(-I, p, p)    zeros(T, p, pq + num_trivars); # upper bound
            Matrix{T}(-I, p + 1, p + 1)    zeros(T, p + 1, pq + num_trivars); # geomean
            zeros(T, pq + p)    G_soc_epi    G_soc    zeros(T, pq + p, num_trivars); # epinormeucl
            ]
        h = vcat(h_nonneg, h_nmax, zeros(2p + 1 + pq))

        return (c = c, A = A, b = b, G = G, h = h, cones = cones, p = p)
    end

    if (rootdet_obj && !use_rootdet) || (logdet_obj && !use_logdet)
        # requires an upper triangular matrix of additional variables, ordered row wise
        num_trivars = CO.svec_length(q)

        # vectorized dimension of psd matrix
        dimvec = q * (2 * q + 1)
        G_psd = zeros(T, dimvec, p + num_trivars)

        # variables in upper triangular matrix numbered row-wise
        diag_idx(i::Int) = (i == 1 ? 1 : 1 + sum(q - j for j in 0:(i - 2)))

        # V*diag(np)*V
        l = 1
        for i in 1:q, j in 1:i
            for k in 1:p
                G_psd[l, k] = -V[i, k] * V[j, k]
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
                G_psd[l, p + tri_idx] = -1
                l += 1
                tri_idx += 1
            end
            # diag(triangle)
            # skip zero-valued elements
            l += i - 1
            G_psd[l, p + diag_idx(i)] = -1
            l += 1
        end
        MU.vec_to_svec!(G_psd, rt2 = sqrt(T(2)))

        h_psd = zeros(T, dimvec)
        push!(cones, CO.PosSemidefTri{T, T}(dimvec))
    end


    if rootdet_obj
        @assert !use_rootdet
        c = vcat(zeros(T, p + num_trivars), -one(T))
        A = [A zeros(T, 1, num_trivars + 1)]
        h_geo = zeros(T, q + 1)
        G_geo = zeros(T, q, num_trivars)
        for i in 1:q
            G_geo[i, sum(1:i)] = -1
        end
        push!(cones, CO.HypoGeomean{T}(fill(inv(T(q)), q)))
        # all conic constraints
        G = [
            Matrix{T}(-I, p, p)    zeros(T, p, num_trivars + 1); # nonneg
            Matrix{T}(-I, p, p)    zeros(T, p, num_trivars + 1); # nmax
            G_psd    zeros(T, dimvec, 1); # psd
            zeros(T, 1, p)    zeros(T, 1, num_trivars)    -one(T); # hypogeomean
            zeros(T, q, p)    G_geo    zeros(T, q); # hypogeomean
            ]
        h = vcat(h_nonneg, h_nmax, h_psd, h_geo)
        return (c = c, A = A, b = b, G = G, h = h, cones = cones, p = p)
    end

    if logdet_obj
        if use_logdet
            @assert use_linops
            A = BlockMatrix{T}(1, p + 1, [A], [1:1], [2:(p + 1)])
            G = BlockMatrix{T}(
                2 * p + 2 + dimvec,
                p + 1,
                [-I, -I, -ones(T, 1, 1), G_detcone],
                [1:p, (p + 1):(2 * p), (2 * p + 1):(2 * p + 1), (2 * p + 3):(2 * p + 2 + dimvec)],
                [2:(p + 1), 2:(p + 1), 1:1, 2:(p + 1)]
                )
            h = vcat(h_nonneg, h_nmax, h_detcone)
        else
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
            # maximize the sum of hypograph variables of all hypoperlog cones
            c = vcat(zeros(T, p + num_trivars), -ones(T, num_hypo))

            if use_sumlog
                G_logvars = zeros(T, q, num_trivars)
                for i in 1:q
                    G_logvars[i, diag_idx(i)] = -1
                end
                G_log = [
                    # hypograph row
                    zeros(T, 1, p + num_trivars)    -one(T)
                    # perspective row
                    zeros(T, 1, dimx)
                    # log row
                    zeros(T, q, p)    G_logvars    zeros(T, q)
                    ]
                h_log = vcat(zero(T), one(T), zeros(T, q))
                push!(cones, CO.HypoPerLog{T}(q + 2))
            else
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
                    push!(cones, CO.HypoPerLog{T}(3))
                    offset += 3
                end
            end
            if use_linops
                A = BlockMatrix{T}(1, p + padx, [A], [1:1], [1:p])
                G = BlockMatrix{T}(
                    2 * p + dimvec + size(G_log, 1),
                    dimx,
                    [-I, -I, G_psd, G_log],
                    [1:p, (p + 1):(2 * p), (2 * p + 1):(2 * p + dimvec), (2 * p + dimvec + 1):(2 * p + dimvec + size(G_log, 1))],
                    [1:p, 1:p, 1:dimx, 1:dimx]
                    )
            else
                # pad with triangle matrix variables and q hypoperlog cone hypograph variables
                A = [A zeros(T, 1, padx)]
                G_nonneg = hcat(Matrix{T}(-I, p, p), zeros(T, p, padx))
                G_nmax = hcat(Matrix{T}(-I, p, p), zeros(T, p, padx))
                # all conic constraints
                G = [
                    G_nonneg;
                    G_nmax;
                    G_psd    zeros(T, dimvec, num_hypo);
                    G_log;
                    ]
            end
            h = vcat(h_nonneg, h_nmax, h_psd, h_log)
        end
    end

    return (c = c, A = A, b = b, G = G, h = h, cones = cones, p = p)
end

expdesign1(T::Type{<:Real}) = expdesign(T, 25, 75, 125, 5, use_logdet = true, logdet_obj = true)
expdesign2(T::Type{<:Real}) = expdesign(T, 10, 30, 50, 5, use_logdet = true, logdet_obj = true)
expdesign3(T::Type{<:Real}) = expdesign(T, 5, 15, 25, 5, use_logdet = true, logdet_obj = true)
expdesign4(T::Type{<:Real}) = expdesign(T, 4, 8, 12, 3, use_logdet = true, logdet_obj = true)
expdesign5(T::Type{<:Real}) = expdesign(T, 3, 5, 7, 2, use_logdet = true, logdet_obj = true)
expdesign6(T::Type{<:Real}) = expdesign(T, 25, 75, 125, 5, use_logdet = false, logdet_obj = true)
expdesign7(T::Type{<:Real}) = expdesign(T, 10, 30, 50, 5, use_logdet = false, logdet_obj = true)
expdesign8(T::Type{<:Real}) = expdesign(T, 5, 15, 25, 5, use_logdet = false, logdet_obj = true)
expdesign9(T::Type{<:Real}) = expdesign(T, 4, 8, 12, 3, use_logdet = false, logdet_obj = true)
expdesign10(T::Type{<:Real}) = expdesign(T, 3, 5, 7, 2, use_logdet = false, logdet_obj = true)
expdesign11(T::Type{<:Real}) = expdesign(T, 4, 8, 12, 3, use_logdet = true, use_linops = true, logdet_obj = true)
expdesign12(T::Type{<:Real}) = expdesign(T, 4, 8, 12, 3, use_logdet = false, use_sumlog = false, use_linops = true, logdet_obj = true)
expdesign13(T::Type{<:Real}) = expdesign(T, 4, 8, 12, 3, use_logdet = false, use_sumlog = false, use_linops = false, logdet_obj = true)
expdesign14(T::Type{<:Real}) = expdesign(T, 4, 8, 12, 3, use_logdet = false, use_sumlog = true, use_linops = true, logdet_obj = true)
expdesign15(T::Type{<:Real}) = expdesign(T, 25, 75, 125, 5, use_rootdet = true, rootdet_obj = true)
expdesign16(T::Type{<:Real}) = expdesign(T, 10, 30, 50, 5, use_rootdet = true, rootdet_obj = true)
expdesign17(T::Type{<:Real}) = expdesign(T, 5, 15, 25, 5, use_rootdet = true, rootdet_obj = true)
expdesign18(T::Type{<:Real}) = expdesign(T, 4, 8, 12, 3, use_rootdet = true, rootdet_obj = true)
expdesign19(T::Type{<:Real}) = expdesign(T, 3, 5, 7, 2, use_rootdet = true, rootdet_obj = true)
expdesign20(T::Type{<:Real}) = expdesign(T, 25, 75, 125, 5, geomean_obj = true)
expdesign21(T::Type{<:Real}) = expdesign(T, 10, 30, 50, 5, geomean_obj = true)
expdesign22(T::Type{<:Real}) = expdesign(T, 5, 15, 25, 5, geomean_obj = true)
expdesign23(T::Type{<:Real}) = expdesign(T, 4, 8, 12, 3, geomean_obj = true)
expdesign24(T::Type{<:Real}) = expdesign(T, 3, 5, 7, 2, geomean_obj = true)
expdesign25(T::Type{<:Real}) = expdesign(T, 25, 75, 125, 5, use_rootdet = false, rootdet_obj = true)
expdesign26(T::Type{<:Real}) = expdesign(T, 10, 30, 50, 5, use_rootdet = false, rootdet_obj = true)
expdesign27(T::Type{<:Real}) = expdesign(T, 5, 15, 25, 5, use_rootdet = false, rootdet_obj = true)
expdesign28(T::Type{<:Real}) = expdesign(T, 4, 8, 12, 3, use_rootdet = false, rootdet_obj = true)
expdesign29(T::Type{<:Real}) = expdesign(T, 3, 5, 7, 2, use_rootdet = false, rootdet_obj = true)

instances_expdesign_all = [
    expdesign1,
    expdesign2,
    expdesign3,
    expdesign4,
    expdesign5,
    expdesign6,
    expdesign7,
    expdesign8,
    expdesign9,
    expdesign10,
    expdesign13,
    expdesign14,
    expdesign15,
    expdesign16,
    expdesign17,
    expdesign18,
    expdesign19,
    expdesign20,
    expdesign21,
    expdesign22,
    expdesign23,
    expdesign24,
    expdesign25,
    expdesign26,
    expdesign27,
    expdesign28,
    expdesign29,
    ]
instances_expdesign_linops = [
    expdesign11,
    expdesign12,
    expdesign14,
]
instances_expdesign_few = [
    expdesign5,
    expdesign10,
    expdesign19,
    expdesign24,
    expdesign29,
    ]

function test_expdesign(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @show r.x[2:(d.p + 1)]
    @test r.status == :Optimal
    return
end

test_expdesign.(instances_expdesign_few)

# [0.5892327034087503, 1.9848818350565813, 2.2756593473710542, 2.1502260256372683, 8.852634136857773e-8]
# [1.9848817473638083, 2.2756593374286678, 2.150225974602002, 9.154905833419491e-8, 23.9316009078927]
# [0.5892325002481775, 1.9848819598634755, 2.2756594566610353, 2.150226071709818, 1.151749451366868e-8]
# [1.4000000006502067, 1.3999999992790264, 1.3999999998436734, 1.4000000000280783, 1.4000000001990138]
