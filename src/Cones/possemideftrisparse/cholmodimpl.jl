#=
NOTE currently we do not restrict the sparsity pattern to be chordal here (at the
cost of not being able to obtain "closed form" hess sqrt and inv hess oracles)

TODO
- improve efficiency of hessian calculations using structure
- use inbounds
- improve efficiency of frontal matrix scattering etc
- maybe allow passing more options to CHOLMOD
=#

import SuiteSparse.CHOLMOD

"""
$(TYPEDEF)

CHOLMOD sparse Cholesky-based implementation for the sparse positive semidefinite
cone [`PosSemidefTriSparse`](@ref). Note only BLAS floating point types are
supported.
"""
struct PSDSparseCholmod <: PSDSparseImpl end

mutable struct PSDSparseCholmodCache{T <: BlasReal, R <: RealOrComplex{T}} <:
    PSDSparseCache{T, R}
    sparse_point
    sparse_point_map
    symb_mat
    supers
    super_map
    parents
    ancestors
    num_cols
    num_rows
    J_rows
    L_idxs
    F_blocks
    L_blocks
    S_blocks
    L_pr_blocks
    S_pr_blocks
    L_pr_pr_blocks
    map_blocks
    temp_blocks
    rel_idxs
    PSDSparseCholmodCache{T, R}() where {T <: BlasReal, R <: RealOrComplex{T}} =
        new{T, R}()
end

function setup_extra_data!(
    cone::PosSemidefTriSparse{PSDSparseCholmod, T, R},
    ) where {T <: BlasReal, R <: RealOrComplex{T}}
    cone.cache = cache = PSDSparseCholmodCache{T, R}()

    # setup symbolic factorization
    side = cone.side
    dim_R = length(cone.row_idxs)

    fake_point = [R(l) for l in 1:dim_R]
    sparse_point = cache.sparse_point = CHOLMOD.Sparse(Hermitian(
        sparse(cone.row_idxs, cone.col_idxs, fake_point, side, side), :L))
    sparse_point_map = cache.sparse_point_map = zeros(Int, dim_R)
    sx_ptr = Ptr{R}(unsafe_load(sparse_point.ptr).x)
    for l in 1:dim_R
        sparse_point_map[Int(unsafe_load(sx_ptr, l))] = l
    end
    CHOLMOD.@cholmod_param supernodal = 2 begin
        cache.symb_mat = CHOLMOD.symbolic(sparse_point, postorder=true)
    end
    symb_mat = cache.symb_mat

    f = unsafe_load(pointer(symb_mat))
    @assert f.n == cone.side
    @assert f.is_super != 0

    num_super = Int(f.nsuper)
    supers = cache.supers = [unsafe_load(f.super, i) + 1 for i in 1:num_super]
    push!(supers, side + 1)
    fs = [unsafe_load(f.s, i) + 1 for i in 1:Int(f.ssize)]
    fpi = [unsafe_load(f.pi, i) + 1 for i in 1:num_super]
    push!(fpi, length(fs) + 1)
    fpx = [unsafe_load(f.px, i) + 1 for i in 1:num_super]

    # construct super_map
    # super_map[k] = s if column k is in supernode s
    super_map = cache.super_map = zeros(Int, side)
    for s in 1:num_super, k in supers[s]:(supers[s + 1] - 1)
        super_map[k] = s
    end

    # construct supernode tree
    parents = cache.parents = zeros(Int, num_super)
    for k in 1:num_super
        nn = supers[k + 1] - supers[k] # n cols
        nj = fpi[k + 1] - fpi[k] # n rows
        na = nj - nn # n rows below tri
        @assert nn >= 0
        @assert nj >= 0
        @assert na >= 0

        if !iszero(na)
            kparloc = fs[fpi[k] + nn]
            kparent = super_map[kparloc]
            @assert supers[kparent] <= kparloc < supers[kparent + 1]
            parents[k] = kparent
        end
    end
    ancestors = cache.ancestors = [Int[] for k in 1:num_super]
    for k in 1:num_super
        curr = k
        while true
            push!(ancestors[k], curr)
            curr = parents[curr]
            if curr == 0
                break
            end
        end
    end

    num_cols = cache.num_cols = Vector{Int}(undef, num_super)
    num_rows = cache.num_rows = Vector{Int}(undef, num_super)
    L_idxs = cache.L_idxs = Vector{UnitRange{Int}}(undef, num_super)
    J_rows = cache.J_rows = Vector{Vector}(undef, num_super)
    F_blocks = cache.F_blocks = Vector{Matrix{R}}(undef, num_super)
    L_blocks = cache.L_blocks = Vector{Matrix{R}}(undef, num_super)
    L_pr_blocks = cache.L_pr_blocks = Vector{Matrix{R}}(undef, num_super)
    L_pr_pr_blocks = cache.L_pr_pr_blocks = Vector{Matrix{R}}(undef, num_super)
    S_blocks = cache.S_blocks = Vector{Matrix{R}}(undef, num_super)
    S_pr_blocks = cache.S_pr_blocks = Vector{Matrix{R}}(undef, num_super)
    temp_blocks = cache.temp_blocks = Vector{Matrix{R}}(undef, num_super)
    rel_idxs = cache.rel_idxs = [Tuple{Int, Int}[] for k in 1:num_super]
    for k in reverse(1:num_super)
        num_col = num_cols[k] = supers[k + 1] - supers[k]
        num_row = num_rows[k] = fpi[k + 1] - fpi[k] # n rows
        @assert 0 < num_col <= num_row

        J_k = J_rows[k] = fs[fpi[k]:(fpi[k + 1] - 1)]
        @assert length(J_k) == num_row

        L_idxs[k] = fpx[k] - 1 .+ (1:(num_row * num_col))

        F_blocks[k] = zeros(R, num_row, num_row)
        num_below = num_row - num_col
        L_blocks[k] = zeros(R, num_row, num_col)
        L_pr_blocks[k] = zeros(R, num_row, num_col)
        L_pr_pr_blocks[k] = zeros(R, num_row, num_col)
        S_blocks[k] = zeros(R, num_below, num_below)
        S_pr_blocks[k] = zeros(R, num_below, num_below)
        temp_blocks[k] = zeros(R, num_row, num_col)

        if num_row > num_col
            # setup relative indices in frontal matrix
            rel_idx = rel_idxs[k]
            k_par = parents[k]
            I_k = J_k[(num_col + 1):end]
            for (idx_i, i) in enumerate(I_k), (idx_j, j) in
                enumerate(J_rows[k_par])
                if i == j
                    push!(rel_idx, (idx_i, idx_j))
                end
            end
        end
    end

    iperm = invperm(symb_mat.p)
    map_blocks = cache.map_blocks = Vector{Tuple{Int, Int, Int, Bool, Bool}}(
        undef, dim_R)
    for i in 1:dim_R
        row = iperm[cone.row_idxs[i]]
        col = iperm[cone.col_idxs[i]]
        if row < col
            (row, col) = (col, row)
            swapped = true
        else
            swapped = false
        end
        super = super_map[col]
        J = J_rows[super]
        row_idx = findfirst(r -> r == row, J)
        col_idx = col - supers[super] + 1
        map_blocks[i] = (super, row_idx, col_idx, row != col, swapped)
    end

    return
end

function update_feas(
    cone::PosSemidefTriSparse{PSDSparseCholmod, T, R},
    ) where {T <: BlasReal, R <: RealOrComplex{T}}
    @assert !cone.feas_updated
    point = cone.point
    cache = cone.cache
    sparse_point = cache.sparse_point

    # svec scale point and copy directly to CHOLDMOD Sparse data structure
    sx_ptr = Ptr{R}(unsafe_load(sparse_point.ptr).x)
    if cone.is_complex
        idx = 1
        @inbounds for (p_idx, s_idx) in enumerate(cache.sparse_point_map)
            if cone.row_idxs[p_idx] == cone.col_idxs[p_idx]
                p = Complex(point[idx])
                idx += 1
            else
                p = Complex(point[idx], point[idx + 1]) / cone.rt2
                idx += 2
            end
            unsafe_store!(sx_ptr, p, s_idx)
        end
    else
        @inbounds for (p_idx, s_idx) in enumerate(cache.sparse_point_map)
            p = point[p_idx]
            if cone.row_idxs[p_idx] != cone.col_idxs[p_idx]
                p /= cone.rt2
            end
            unsafe_store!(sx_ptr, p, s_idx)
        end
    end

    # update numeric factorization
    CHOLMOD.@cholmod_param supernodal = 2 begin
        cone.is_feas = isposdef(CHOLMOD.cholesky!(cache.symb_mat, sparse_point;
            check = false))
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(
    cone::PosSemidefTriSparse{PSDSparseCholmod, T, R},
    ) where {T <: BlasReal, R <: RealOrComplex{T}}
    @assert cone.is_feas
    cache = cone.cache
    num_cols = cache.num_cols
    num_rows = cache.num_rows

    # update L blocks from CHOLMOD numerical factorization
    lx_ptr = Ptr{R}(unsafe_load(pointer(cache.symb_mat)).x)
    @inbounds for k in 1:length(num_cols)
        L_block = cache.L_blocks[k]
        for (l, lx_idx) in enumerate(cache.L_idxs[k])
            L_block[l] = unsafe_load(lx_ptr, lx_idx)
        end
    end

    # build inv blocks
    @inbounds for k in reverse(1:length(num_cols))
        num_col = num_cols[k]
        num_row = num_rows[k]
        idxs_n = 1:num_col
        F_block = cache.F_blocks[k]
        L_block = cache.L_blocks[k]

        @views L_n = L_block[idxs_n, :]
        @views F_nn = F_block[idxs_n, idxs_n]
        copyto!(F_nn, L_n)
        LAPACK.potri!('L', F_nn)

        if num_row > num_col
            idxs_a = (num_col + 1):num_row
            @views F_aa = Hermitian(F_block[idxs_a, idxs_a], :L)
            @views F_an = F_block[idxs_a, idxs_n]
            @views L_a = rdiv!(L_block[idxs_a, :], LowerTriangular(L_n))

            F_aa.data .= 0
            F_par = Hermitian(cache.F_blocks[cache.parents[k]], :L)
            rel_idx = cache.rel_idxs[k]
            for (i, j) in rel_idx, (i2, j2) in rel_idx # TODO only lower tri
                F_aa.data[i, i2] = F_par[j, j2]
            end

            mul!(F_an, F_aa, L_a, -1, false)
            mul!(F_nn, F_an', L_a, -1, true)

            copyto!(cache.S_blocks[k], F_aa.data) # use in Hessian calculations
        end

        @views cache.temp_blocks[k] = F_block[:, idxs_n]
    end

    smat_to_svec_sparse!(cone.grad, cache.temp_blocks, cone)
    cone.grad .*= -1

    cone.grad_updated = true
    return cone.grad
end

# for each column in identity, get hess prod to build explicit hess
function update_hess(cone::PosSemidefTriSparse{PSDSparseCholmod})
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    cache = cone.cache
    temp_blocks = cache.temp_blocks
    rt2 = cone.rt2
    invrt2 = inv(rt2)
    H = cone.hess.data

    H_idx_j = 1
    @inbounds for (j, (super_j, row_idx_j, col_idx_j, scal_j, swapped_j)) in
        enumerate(cache.map_blocks)
        for H_block in temp_blocks
            H_block .= 0
        end

        if cone.is_complex
            # real part j
            temp_blocks[super_j][row_idx_j, col_idx_j] = (scal_j ? invrt2 : 1)
            _hess_step1(cone, cache.ancestors[super_j])
            _hess_step2(cone, cache.ancestors[super_j], false)
            out_blocks = _hess_step3(cone)
            H_idx_i = 1
            for i in 1:j
                (super_i, row_idx_i, col_idx_i, scal_i, swapped_i) =
                    cache.map_blocks[i]
                HijR = out_blocks[super_i][row_idx_i, col_idx_i]
                if scal_i
                    HijR *= rt2
                end
                H[H_idx_i, H_idx_j] = real(HijR) # real part i
                H_idx_i += 1
                if row_idx_i != col_idx_i
                    # complex part i
                    H[H_idx_i, H_idx_j] = swapped_i ? -imag(HijR) : imag(HijR)
                    H_idx_i += 1
                end
            end
            H_idx_j += 1

            if row_idx_j != col_idx_j
                # complex part j
                for H_block in temp_blocks
                    H_block .= 0
                end
                temp_blocks[super_j][row_idx_j, col_idx_j] =
                    (scal_j ? invrt2 : 1) * im
                _hess_step1(cone, cache.ancestors[super_j])
                _hess_step2(cone, cache.ancestors[super_j], false)
                out_blocks = _hess_step3(cone)
                H_idx_i = 1
                for i in 1:j
                    (super_i, row_idx_i, col_idx_i, scal_i, swapped_i) =
                        cache.map_blocks[i]
                    HijR = out_blocks[super_i][row_idx_i, col_idx_i]
                    if scal_i
                        HijR *= rt2
                    end
                    # real part i
                    H[H_idx_i, H_idx_j] = swapped_j ? -real(HijR) : real(HijR)
                    H_idx_i += 1
                    if row_idx_i != col_idx_i
                        # complex part i
                        H[H_idx_i, H_idx_j] = xor(swapped_i, swapped_j) ?
                            -imag(HijR) : imag(HijR)
                        H_idx_i += 1
                    end
                end
                H_idx_j += 1
            end
        else
            temp_blocks[super_j][row_idx_j, col_idx_j] = (scal_j ? invrt2 : 1)
            _hess_step1(cone, cache.ancestors[super_j])
            _hess_step2(cone, cache.ancestors[super_j], false)
            out_blocks = _hess_step3(cone)

            for i in 1:j
                (super_i, row_idx_i, col_idx_i, scal_i, swapped_i) =
                    cache.map_blocks[i]
                H[i, j] = out_blocks[super_i][row_idx_i, col_idx_i]
                if scal_i
                    H[i, j] *= rt2
                end
            end
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod_slow!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::PosSemidefTriSparse,
    )
    cone.use_hess_prod_slow_updated || update_use_hess_prod_slow(cone)
    @assert cone.hess_updated
    cone.use_hess_prod_slow || return hess_prod!(prod, arr, cone)
    @assert cone.grad_updated
    cache = cone.cache
    temp_blocks = cache.temp_blocks

    @inbounds for i in 1:size(arr, 2)
        @views svec_to_smat_sparse!(temp_blocks, arr[:, i], cone)
        _hess_step1(cone, eachindex(cache.num_cols))
        _hess_step2(cone, eachindex(cache.num_cols), false)
        _hess_step3(cone)
        @views smat_to_svec_sparse!(prod[:, i], temp_blocks, cone)
    end

    return prod
end

function _hess_step1(
    cone::PosSemidefTriSparse{PSDSparseCholmod},
    supernode_list::AbstractVector{Int},
    )
    @assert cone.grad_updated
    cache = cone.cache
    temp_blocks = cache.temp_blocks

    @inbounds for k in supernode_list
        num_col = cache.num_cols[k]
        num_row = cache.num_rows[k]
        idxs_n = 1:num_col
        idxs_a = (num_col + 1):num_row
        F_block = cache.F_blocks[k]
        temp_block = temp_blocks[k]

        @views F_block[idxs_a, idxs_a] .= 0
        @views F_block[:, idxs_n] .= temp_block
    end

    @inbounds for k in supernode_list
        num_col = cache.num_cols[k]
        num_row = cache.num_rows[k]
        idxs_n = 1:num_col
        F_block = cache.F_blocks[k]
        temp_block = temp_blocks[k]

        if num_row > num_col
            outer_L_prod(cone, k)
            idxs_a = (num_col + 1):num_row
            @views F_aa = Hermitian(F_block[idxs_a, idxs_a], :L)

            F_par = cache.F_blocks[cache.parents[k]]
            rel_idx = cache.rel_idxs[k]
            for (i, j) in rel_idx, (i2, j2) in rel_idx # TODO only lower tri
                F_par[j, j2] += F_aa[i, i2]
            end
        end

        @views copyto!(temp_block, F_block[:, idxs_n])
    end

    return temp_blocks
end

function _hess_step2(
    cone::PosSemidefTriSparse{PSDSparseCholmod},
    supernode_list::AbstractVector{Int},
    save_L_pr::Bool,
    )
    cache = cone.cache
    temp_blocks = cache.temp_blocks

    @inbounds for k in supernode_list
        num_col = cache.num_cols[k]
        num_row = cache.num_rows[k]
        idxs_n = 1:num_col
        temp_block = temp_blocks[k]
        L_pr_block = cache.L_pr_blocks[k]

        @views temp_block_n = temp_block[idxs_n, :]
        copytri!(temp_block_n, 'L', true)
        @views L_n = LowerTriangular(cache.L_blocks[k][idxs_n, :])
        if save_L_pr
            @views copyto!(L_pr_block[idxs_n, :], temp_block_n)
        end
        ldiv!(L_n, temp_block_n)
        ldiv!(L_n', temp_block_n)
        rdiv!(temp_block, L_n')
        rdiv!(temp_block, L_n)

        if num_row > num_col
            idxs_a = (num_col + 1):num_row
            @views temp_block_a = temp_block[idxs_a, :]
            if save_L_pr
                @views copyto!(L_pr_block[idxs_a, :], temp_block_a)
            end
            @views F_an = cache.F_blocks[k][idxs_a, idxs_n]
            mul!(F_an, Hermitian(cache.S_blocks[k], :L), temp_block_a)
            copyto!(temp_block_a, F_an)
        end
    end

    return temp_blocks
end

function _hess_step3(
    cone::PosSemidefTriSparse{PSDSparseCholmod},
    )
    cache = cone.cache
    temp_blocks = cache.temp_blocks

    @inbounds for k in reverse(1:length(cache.num_cols))
        num_col = cache.num_cols[k]
        num_row = cache.num_rows[k]
        idxs_n = 1:num_col
        F_block = cache.F_blocks[k]
        temp_block = temp_blocks[k]
        @. @views F_block[:, idxs_n] = temp_block
        if num_row > num_col
            idxs_a = (num_col + 1):num_row
            @views L_a = cache.L_blocks[k][idxs_a, :]
            @views F_aa = Hermitian(F_block[idxs_a, idxs_a], :L)
            @views F_an = F_block[idxs_a, idxs_n]
            @views F_nn = Hermitian(F_block[idxs_n, idxs_n], :L)
            F_aa.data .= 0
            F_par = Hermitian(cache.F_blocks[cache.parents[k]], :L)
            rel_idx = cache.rel_idxs[k]
            for (i, j) in rel_idx, (i2, j2) in rel_idx # TODO only lower tri
                F_aa.data[i, i2] = F_par[j, j2]
            end
            mul!(F_nn.data, F_an', L_a, -1, true)
            mul!(F_an, F_aa, L_a, -1, true)
            mul!(F_nn.data, L_a', F_an, -1, true)
        end
        @views copyto!(temp_block, F_block[:, idxs_n])
    end

    return temp_blocks
end

function outer_L_prod(
    cone::PosSemidefTriSparse{PSDSparseCholmod},
    k::Int,
    )
    cache = cone.cache

    num_col = cache.num_cols[k]
    num_row = cache.num_rows[k]
    idxs_n = 1:num_col
    F_block = cache.F_blocks[k]
    idxs_a = (num_col + 1):num_row
    @views L_a = cache.L_blocks[k][idxs_a, :]
    @views F_aa = Hermitian(F_block[idxs_a, idxs_a], :L)
    @views F_an = F_block[idxs_a, idxs_n]
    @views F_nn = Hermitian(F_block[idxs_n, idxs_n], :L)
    mul!(F_aa.data, L_a, F_an', -1, true)
    mul!(F_an, L_a, F_nn, -1, true)
    mul!(F_aa.data, F_an, L_a', -1, true)

    return F_block
end

function dder3(
    cone::PosSemidefTriSparse{PSDSparseCholmod},
    dir::AbstractVector,
    )
    @assert cone.grad_updated
    cache = cone.cache
    temp_blocks = cache.temp_blocks

    @views svec_to_smat_sparse!(temp_blocks, dir, cone)
    _hess_step1(cone, eachindex(cache.num_cols))
    _hess_step2(cone, eachindex(cache.num_cols), true)
    _hess_step3(cone)
    @inbounds for k in eachindex(cache.num_cols)
        idxs_a = (cache.num_cols[k] + 1):cache.num_rows[k]
        F_block = cache.F_blocks[k]
        @. @views cache.S_pr_blocks[k] = F_block[idxs_a, idxs_a]
        F_block .= 0
    end

    @inbounds for k in eachindex(cache.num_cols)
        num_col = cache.num_cols[k]
        num_row = cache.num_rows[k]
        idxs_n = 1:num_col
        idxs_a = (num_col + 1):num_row
        F_block = cache.F_blocks[k]
        @views L_n = LowerTriangular(cache.L_blocks[k][idxs_n, :])
        @views L_pr = cache.L_pr_blocks[k][idxs_a, :]
        @views temp_block_a = temp_blocks[k][idxs_a, :]

        if num_row > num_col
            outer_L_prod(cone, k)
            @views F_aa = F_block[idxs_a, idxs_a]
            mul!(temp_block_a, L_pr, L_n)
            mul!(F_aa, temp_block_a, temp_block_a', -2, true)
            F_par = cache.F_blocks[cache.parents[k]]
            rel_idx = cache.rel_idxs[k]
            for (i, j) in rel_idx, (i2, j2) in rel_idx
                F_par[j, j2] += F_aa[i, i2]
            end
        end

        # transform block from linearized factorization into
        # block for linearized inverse
        S = cache.S_blocks[k]
        L_pr_pr = cache.L_pr_pr_blocks[k]
        @views D_pr = cache.L_pr_blocks[k][idxs_n, :]
        @views temp_block_n = temp_blocks[k][idxs_n, :]
        @views copyto!(L_pr_pr, F_block[:, idxs_n])
        @views L_pr_pr_a = L_pr_pr[idxs_a, :]
        mul!(L_pr_pr_a, L_pr, D_pr, -2, true)
        rdiv!(L_pr_pr_a, L_n')
        rdiv!(L_pr_pr_a, L_n)
        @. @views temp_block_n = -L_pr_pr[idxs_n, :]
        ldiv!(L_n, D_pr)
        mul!(temp_block_n, D_pr', D_pr, 2, true)
        ldiv!(L_n, temp_block_n)
        ldiv!(L_n', temp_block_n)
        rdiv!(temp_block_n, L_n')
        rdiv!(temp_block_n, L_n)
        mul!(temp_block_a, S, L_pr)
        mul!(temp_block_n, L_pr', temp_block_a, 2, true)
        mul!(temp_block_a, cache.S_pr_blocks[k], L_pr, 2, false)
        mul!(temp_block_a, S, L_pr_pr_a, -1, true)
    end

    _hess_step3(cone)
    smat_to_svec_sparse!(cone.dder3, cache.temp_blocks, cone)
    cone.dder3 ./= 2

    return cone.dder3
end

function svec_to_smat_sparse!(
    blocks::Vector{Matrix{T}},
    vec::AbstractVector{T},
    cone::PosSemidefTriSparse{PSDSparseCholmod, T, T},
    ) where {T <: BlasReal}
    @assert length(vec) == length(cone.row_idxs)
    for b in blocks
        b .= 0
    end
    @inbounds for (i, (super, row_idx, col_idx, scal, swapped)) in enumerate(cone.cache.map_blocks)
        vec_i = vec[i]
        if scal
            vec_i /= cone.rt2
        end
        blocks[super][row_idx, col_idx] = vec_i
    end
    return blocks
end

function smat_to_svec_sparse!(
    vec::AbstractVector{T},
    blocks::Vector{Matrix{T}},
    cone::PosSemidefTriSparse{PSDSparseCholmod, T, T},
    ) where {T <: BlasReal}
    @assert length(vec) == length(cone.row_idxs)
    @inbounds for (i, (super, row_idx, col_idx, scal, swapped)) in enumerate(cone.cache.map_blocks)
        vec_i = blocks[super][row_idx, col_idx]
        if scal
            vec_i *= cone.rt2
        end
        vec[i] = vec_i
    end
    return vec
end

function svec_to_smat_sparse!(
    blocks::Vector{Matrix{Complex{T}}},
    vec::AbstractVector{T},
    cone::PosSemidefTriSparse{PSDSparseCholmod, T, Complex{T}},
    ) where {T <: BlasReal}
    for b in blocks
        b .= 0
    end
    idx = 1
    @inbounds for (i, (super, row_idx, col_idx, scal, swapped)) in enumerate(cone.cache.map_blocks)
        if scal
            blocks[super][row_idx, col_idx] = Complex(vec[idx], (swapped ? -vec[idx + 1] : vec[idx + 1])) / cone.rt2
            idx += 2
        else
            blocks[super][row_idx, col_idx] = vec[idx]
            idx += 1
        end
    end
    @assert idx == length(vec) + 1
    return blocks
end

function smat_to_svec_sparse!(
    vec::AbstractVector{T},
    blocks::Vector{Matrix{Complex{T}}},
    cone::PosSemidefTriSparse{PSDSparseCholmod, T, Complex{T}},
    ) where {T <: BlasReal}
    idx = 1
    @inbounds for (i, (super, row_idx, col_idx, scal, swapped)) in enumerate(cone.cache.map_blocks)
        vec_R = blocks[super][row_idx, col_idx]
        if scal
            vec_R *= cone.rt2
            vec[idx] = real(vec_R)
            vec[idx + 1] = swapped ? -imag(vec_R) : imag(vec_R)
            idx += 2
        else
            vec[idx] = real(vec_R)
            idx += 1
        end
    end
    @assert idx == length(vec) + 1
    return vec
end
