#=
see description in JuMP.jl
=#

struct DOptimalDesignNative{T <: Real} <: ExampleInstanceNative{T}
    q::Int
    p::Int
    n::Int
    n_max::Int
    logdet_obj::Bool
    rootdet_obj::Bool
    geomean_obj::Bool
    use_logdet::Bool # use logdet cone, else use extended formulation
    use_rootdet::Bool # use rootdet cone, else use extended formulation
    use_epinorminf::Bool # use infinity norm cone, else use extended formulation
end

function build(inst::DOptimalDesignNative{T}) where {T <: Real}
    (q, p, n, n_max) = (inst.q, inst.p, inst.n, inst.n_max)
    @assert (p > q) && (n > q) && (n_max <= n)
    @assert +(inst.logdet_obj, inst.geomean_obj, inst.rootdet_obj) == 1
    V = randn(T, q, p)

    # upper bound and nonnegativity of numbers of trials
    if inst.use_epinorminf
        h_norminf = vcat(T(n_max) / 2, fill(-T(n_max) / 2, p))
        G_norminf = vcat(zeros(T, 1, p), -Matrix{T}(I, p, p))
        cones = Cones.Cone{T}[Cones.EpiNormInf{T, T}(p + 1)]
    else
        h_norminf = vcat(zeros(T, p), fill(T(n_max), p))
        G_norminf = vcat(Matrix{T}(-I, p, p), Matrix{T}(I, p, p))
        cones = Cones.Cone{T}[Cones.Nonnegative{T}(p), Cones.Nonnegative{T}(p)]
    end

    # constraint on total number of trials
    A = ones(T, 1, p)
    b = T[n]

    if (inst.logdet_obj && inst.use_logdet) ||
        (inst.rootdet_obj && inst.use_rootdet)
        # maximize the hypograph variable of the cone
        c = vcat(-one(T), zeros(T, p))

        # pad with hypograph variable
        A = hcat(zero(T), A)
        G_norminf = hcat(zeros(T, size(G_norminf, 1)), G_norminf)

        # dimension of vectorized matrix V*diag(np)*V'
        dimvec = Cones.svec_length(q)
        G_detcone = zeros(T, dimvec, p)
        l = 1
        for i in 1:q, j in 1:i
            for k in 1:p
                G_detcone[l, k] = -V[i, k] * V[j, k]
            end
            l += 1
        end
        Cones.scale_svec!(G_detcone, sqrt(T(2)))
        @assert l - 1 == dimvec

        if inst.logdet_obj
            push!(cones, Cones.HypoPerLogdetTri{T, T}(dimvec + 2))
            # pad with hypograph variable and perspective variable
            h_detcone = vcat(zero(T), one(T), zeros(T, dimvec))
            # include perspective variable
            G_detcone = [
                -one(T)    zeros(T, 1, p);
                zeros(T, 1, p + 1);
                zeros(T, dimvec)    G_detcone;
                ]
        else
            push!(cones, Cones.HypoRootdetTri{T, T}(dimvec + 1))
            # pad with hypograph variable
            h_detcone = zeros(T, dimvec + 1)
            # include perspective variable
            G_detcone = [
                -one(T)    zeros(T, 1, p);
                zeros(T, dimvec)    G_detcone;
                ]
        end

        # all conic constraints
        G = vcat(G_norminf, G_detcone)
        h = vcat(h_norminf, h_detcone)
    end

    if inst.geomean_obj
        # auxiliary matrix variable has pq elements stored row-major, auxiliary
        # lower tri variable has svec_length(q) elements, also stored row-major
        pq = p * q
        qq = q ^ 2
        num_trivars = Cones.svec_length(q)
        c = vcat(-one(T), zeros(T, p + pq + num_trivars))

        A_VW = zeros(T, qq, pq)
        A_lowertri = zeros(T, qq, num_trivars)
        # rows index (i, j)
        row_idx = 1
        for i in 1:q
            col_offset = sum(1:(i - 1)) + 1
            A_lowertri[row_idx:(row_idx + i - 1),
                col_offset:(col_offset + i - 1)] = -Matrix{T}(-I, i, i)
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
            zero(T)    A    zeros(T, 1, pq + num_trivars);
            zeros(T, qq, 1 + p)    A_VW    A_lowertri;
            ]
        b = vcat(b, zeros(T, qq))

        G_geo = zeros(T, q, num_trivars)
        for i in 1:q
            G_geo[i, sum(1:i)] = -1
        end
        push!(cones, Cones.HypoGeoMean{T}(1 + q))
        G_soc_epi = zeros(T, pq + p, p)
        G_soc = zeros(T, pq + p, pq)
        epi_idx = 1
        col_idx = 1
        for i in 1:p
            push!(cones, Cones.EpiNormEucl{T}(q + 1))
            G_soc_epi[epi_idx, i] = -sqrt(T(q))
            G_soc[(epi_idx + 1):(epi_idx + q), col_idx:(col_idx + q - 1)] =
                Matrix{T}(-I, q, q)
            epi_idx += q + 1
            col_idx += q
        end
        zero1 = zeros(T, size(G_norminf, 1), pq + num_trivars)
        zero2 = zeros(T, pq + p, num_trivars)
        G = [
            zeros(T, size(G_norminf, 1))    G_norminf    zero1;
            -one(T)    zeros(T, 1, p + pq + num_trivars); # geomean
            zeros(T, q, 1 + p + pq)    G_geo; # geomean
            zeros(T, pq + p)    G_soc_epi    G_soc    zero2; # epinormeucl
            ]
        h = vcat(h_norminf, zeros(T, p + 1 + q + pq))
    end

    if (inst.rootdet_obj && !inst.use_rootdet) ||
        (inst.logdet_obj && !inst.use_logdet)
        # extended formulations require an upper tri matrix of additional vars
        # we will store this matrix row-major
        num_trivars = Cones.svec_length(q)
        # vectorized dimension of extended psd matrix
        dimvec = q * (2 * q + 1)
        G_psd = zeros(T, dimvec, p + num_trivars)

        # index of diagonal elements in upper triangular matrix
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
        Cones.scale_svec!(G_psd, sqrt(T(2)))

        h_psd = zeros(T, dimvec)
        push!(cones, Cones.PosSemidefTri{T, T}(dimvec))
    end

    if inst.rootdet_obj && !inst.use_rootdet
        c = vcat(zeros(T, p + num_trivars), -one(T))
        A = hcat(A, zeros(T, 1, num_trivars + 1))
        h_geo = zeros(T, q + 1)
        G_geo = zeros(T, q, num_trivars)
        for i in 1:q
            G_geo[i, diag_idx(i)] = -1
        end
        push!(cones, Cones.HypoGeoMean{T}(1 + q))
        # all conic constraints
        G = [
            G_norminf   zeros(T, size(G_norminf, 1), num_trivars + 1);
            G_psd    zeros(T, dimvec, 1); # psd
            zeros(T, 1, p)    zeros(T, 1, num_trivars)    -one(T); # hypogeomean
            zeros(T, q, p)    G_geo    zeros(T, q); # hypogeomean
            ]
        h = vcat(h_norminf, h_psd, h_geo)
    end

    if inst.logdet_obj && !inst.use_logdet
        # extended formulation for logdet
        # number of experiments, upper triangular matrix, hypograph variables
        dimx = p + num_trivars + q
        padx = num_trivars + q
        num_hypo = q
        # maximize the sum of hypograph variables of all hypoperlog cones
        c = vcat(zeros(T, p + num_trivars), -ones(T, num_hypo))

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
            push!(cones, Cones.HypoPerLog{T}(3))
            offset += 3
        end

        # pad with triangle matrix variables and q hypoperlog cone hypograph variables
        A = [A zeros(T, 1, padx)]
        # all conic constraints
        G = [
            G_norminf    zeros(T, size(G_norminf, 1), padx);
            G_psd    zeros(T, dimvec, num_hypo);
            G_log;
            ]
        h = vcat(h_norminf, h_psd, h_log)
    end

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end
