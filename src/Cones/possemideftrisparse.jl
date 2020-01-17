#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

sparse lower triangle of positive semidefinite matrix cone (unscaled "smat" form)
W \in S^n : 0 >= eigmin(W)

NOTE on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector

dual is sparse PSD completable

TODO
- describe
- hermitian case
- reference
- doesn't seem to need to be chordal/filled
=#

import SuiteSparse.CHOLMOD

mutable struct PosSemidefTriSparse{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    row_idxs::Vector{Int}
    col_idxs::Vector{Int}
    is_complex::Bool
    point::Vector{T}
    rt2::T
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    pointR::Vector{R}
    symb_mat
    fact_mat
    perm
    iperm
    supers
    fpx
    xptr
    supermap
    parents # TODO don't save if not used
    children
    num_cols
    num_rows
    J_rows
    L_blocks
    V_blocks
    inv_blocks
    map_blocks

    function PosSemidefTriSparse{T, R}(
        side::Int,
        row_idxs::Vector{Int},
        col_idxs::Vector{Int},
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T), # TODO get inverse hessian directly
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        num_nz = length(row_idxs)
        @assert length(col_idxs) == num_nz
        @assert all(col_idxs .<= row_idxs .<= side) # TODO improve efficiency
        cone = new{T, R}()
        if R <: Real
            cone.dim = num_nz
            cone.is_complex = false
        else
            tril_dim = num_nz - side
            cone.dim = side + 2 * tril_dim
            cone.is_complex = true
        end
        @assert cone.dim >= 1
        # TODO check diagonals are all present. maybe diagonals go first in point, then remaining elements are just the nonzeros off diag, and row/col idxs are only the off diags
        cone.use_dual = is_dual
        cone.side = side # side dimension of sparse matrix
        cone.row_idxs = row_idxs
        cone.col_idxs = col_idxs
        cone.rt2 = sqrt(T(2))
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

PosSemidefTriSparse{T, R}(side::Int, row_idxs::Vector{Int}, col_idxs::Vector{Int}) where {R <: RealOrComplex{T}} where {T <: Real} = PosSemidefTriSparse{T, R}(side, row_idxs, col_idxs, false)

# reset_data(cone::PosSemidefTriSparse) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function setup_data(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    setup_symbfact(cone)
    return
end

# setup symbolic factorization
function setup_symbfact(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    side = cone.side
    dimR = length(cone.row_idxs)

    cm = CHOLMOD.defaults(CHOLMOD.common_struct)
    unsafe_store!(CHOLMOD.common_print[], 0)
    unsafe_store!(CHOLMOD.common_postorder[], 1)
    # unsafe_store!(CHOLMOD.common_final_ll[], 1)
    unsafe_store!(CHOLMOD.common_supernodal[], 2)

    cone.pointR = ones(R, dimR)
    mat = CHOLMOD.Sparse(Hermitian(sparse(cone.row_idxs, cone.col_idxs, cone.pointR, side, side), :L))
    symb_mat = cone.symb_mat = CHOLMOD.fact_(mat, cm)

    cone.perm = symb_mat.p
    cone.iperm = invperm(cone.perm)

    f = unsafe_load(pointer(symb_mat))
    @assert f.n == cone.side
    @assert f.is_ll != 0
    @assert f.is_super != 0

    num_super = Int(f.nsuper)
    ssize = Int(f.ssize)
    supers = f.super
    fpi = f.pi
    fpx = f.px
    fs = f.s
    fx = f.x

    supers = cone.supers = [unsafe_load(supers, i) + 1 for i in 1:num_super]
    fpi = [unsafe_load(fpi, i) + 1 for i in 1:num_super]
    fpx = cone.fpx = [unsafe_load(fpx, i) + 1 for i in 1:num_super]
    fs = [unsafe_load(fs, i) + 1 for i in 1:ssize]
    cone.xptr = f.x

    push!(supers, side + 1)
    push!(fpi, length(fs) + 1)

    # construct supermap
    # supermap[k] = s if column k is in supernode s
    supermap = cone.supermap = zeros(Int, side)
    for s in 1:num_super, k in supers[s]:(supers[s + 1] - 1)
        supermap[k] = s
    end

    # construct supernode tree
    parents = cone.parents = zeros(Int, num_super)
    children = cone.children = [Int[] for k in 1:num_super]
    for k in 1:num_super
        nn = supers[k + 1] - supers[k] # n cols
        nj = fpi[k + 1] - fpi[k] # n rows
        na = nj - nn # n rows below tri
        @assert nn >= 0
        @assert nj >= 0
        @assert na >= 0

        if !iszero(na)
            kparloc = fs[fpi[k] + nn]
            kparent = supermap[kparloc]
            # @assert k < kparent <= num_super
            @assert supers[kparent] <= kparloc < supers[kparent + 1]
            parents[k] = kparent
            push!(children[kparent], k)
        end
    end

    num_cols = cone.num_cols = Vector{Int}(undef, num_super)
    num_rows = cone.num_rows = Vector{Int}(undef, num_super)
    J_rows = cone.J_rows = Vector{Vector}(undef, num_super)
    cone.L_blocks = Vector{Matrix{R}}(undef, num_super)
    cone.V_blocks = Vector{Matrix{R}}(undef, num_super)
    cone.inv_blocks = Vector{Matrix{R}}(undef, num_super)
    for k in 1:num_super
        num_col = num_cols[k] = supers[k + 1] - supers[k]
        num_row = num_rows[k] = fpi[k + 1] - fpi[k] # n rows
        @assert 0 < num_col <= num_row

        Jk = J_rows[k] = fs[fpi[k]:(fpi[k + 1] - 1)]
        @assert length(Jk) == num_row

        cone.inv_blocks[k] = zeros(R, num_row, num_col)
    end

    map_blocks = cone.map_blocks = Vector{Tuple{Int, Int, Int, Bool, Bool}}(undef, dimR)
    for i in 1:dimR
        row = cone.iperm[cone.row_idxs[i]]
        col = cone.iperm[cone.col_idxs[i]]
        if row < col
            (row, col) = (col, row)
            swapped = true
        else
            swapped = false
        end
        super = supermap[col]
        J = J_rows[super]
        row_idx = findfirst(r -> r == row, J)
        col_idx = col - supers[super] + 1
        map_blocks[i] = (super, row_idx, col_idx, row != col, swapped)
    end

    return
end

get_nu(cone::PosSemidefTriSparse) = cone.side

# real
function set_initial_point(arr::AbstractVector, cone::PosSemidefTriSparse{T, T}) where {T <: Real}
    # set diagonal elements to 1
    for i in eachindex(cone.pointR)
        if cone.row_idxs[i] == cone.col_idxs[i]
            arr[i] = 1
        else
            arr[i] = 0
        end
    end
    return arr
end

# complex
function set_initial_point(arr::AbstractVector, cone::PosSemidefTriSparse{T, Complex{T}}) where {T <: Real}
    # set diagonal elements to 1
    idx = 1
    for i in eachindex(cone.pointR)
        if cone.row_idxs[i] == cone.col_idxs[i]
            arr[idx] = 1
            idx += 1
        else
            arr[idx] = 0
            idx += 1
            arr[idx] = 0
            idx += 1
        end
    end
    return arr
end

function update_feas(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert !cone.feas_updated
    point = cone.point
    pointR = cone.pointR

    # svec scale point
    if cone.is_complex
        idx = 1
        for i in eachindex(pointR)
            if cone.row_idxs[i] == cone.col_idxs[i]
                pointR[i] = point[idx]
                idx += 1
            else
                pointR[i] = Complex(point[idx], point[idx + 1]) / cone.rt2
                idx += 2
            end
        end
    else
        copyto!(pointR, point)
        for i in eachindex(pointR)
            if cone.row_idxs[i] != cone.col_idxs[i]
                pointR[i] /= cone.rt2
            end
        end
    end

    # TODO make more efficient
    sparse_point = sparse(cone.row_idxs, cone.col_idxs, pointR, cone.side, cone.side)
    mat = CHOLMOD.Sparse(Hermitian(sparse_point, :L))
    # compute numeric factorization
    cone.fact_mat = CHOLMOD.cholesky!(cone.symb_mat, mat; check = false)
    cone.is_feas = isposdef(cone.fact_mat)

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.is_feas
    num_super = length(cone.supers) - 1
    children = cone.children
    # parents = cone.parents
    num_cols = cone.num_cols
    num_rows = cone.num_rows
    J_rows = cone.J_rows
    L_blocks = cone.L_blocks
    V_blocks = cone.V_blocks
    inv_blocks = cone.inv_blocks

    # update L blocks from numerical factorization
    f = unsafe_load(pointer(cone.symb_mat))
    for k in 1:num_super
        num_col = num_cols[k]
        num_row = num_rows[k]
        x_idxs = cone.fpx[k] - 1 .+ (1:(num_row * num_col))
        L_blocks[k] = reshape([unsafe_load(f.x, i) for i in x_idxs], num_row, num_col)
    end

    # build inv blocks
    for k in reverse(1:num_super)
        num_col = num_cols[k]
        num_row = num_rows[k]
        idxs_n = 1:num_col
        J = J_rows[k]
        L_block = L_blocks[k]

        @views FD_nn = LowerTriangular(L_block[idxs_n, :])
        FD_inv = inv(FD_nn)

        F = zeros(R, num_row, num_row)
        @views F_nn = Hermitian(F[idxs_n, idxs_n], :L)
        mul!(F_nn.data, FD_inv', FD_inv)

        if num_row > num_col
            idxs_a = (num_col + 1):num_row
            @views F_aa = Hermitian(F[idxs_a, idxs_a], :L)
            @views F_an = F[idxs_a, idxs_n]
            L_a = @views rdiv!(L_block[idxs_a, :], FD_nn)

            V = V_blocks[k]
            copyto!(F_aa.data, V)

            mul!(F_an, V, L_a, -1, false)
            mul!(F_nn.data, F_an', L_a, -1, true)
        end

        # TODO not very efficient since zeros in V
        for k2 in children[k]
            A2 = J_rows[k2][(num_cols[k2] + 1):end]
            EJA2 = Bool[i == j for i in J, j in A2]
            V_blocks[k2] = Hermitian(EJA2' * Hermitian(F, :L) * EJA2, :L)
        end

        inv_blocks[k] = F[:, idxs_n]
    end

    g = cone.grad
    @show cone.map_blocks
    idx = 1
    for (i, (super, row_idx, col_idx, scal, swapped)) in enumerate(cone.map_blocks)
        giR = -inv_blocks[super][row_idx, col_idx]
        if scal
            giR *= cone.rt2
        end
        if cone.is_complex
            g[idx] = real(giR)
            idx += 1
            if scal
                g[idx] = swapped ? -imag(giR) : imag(giR)
                idx += 1
            end
        else
            g[i] = giR
        end
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.grad_updated
    H = cone.hess.data
    rt2 = sqrt(2)
    invrt2 = inv(rt2)

    # for each column in identity, get hess prod to build explicit hess
    # TODO not very efficient way to do the hessian explicitly, but somewhat efficient for hess_prod
    for j in 1:cone.dim
        in_blocks = [copy(L_block) for L_block in cone.L_blocks]
        for H_block in in_blocks
            H_block .= 0
        end
        (super, row_idx, col_idx, scal1, swapped1) = cone.map_blocks[j]
        @assert row_idx >= col_idx
        in_blocks[super][row_idx, col_idx] = (scal1 ? invrt2 : 1)

        out_blocks = H_prod_col(cone, in_blocks)
        for i in 1:j
            (super, row_idx, col_idx, scal2, swapped2) = cone.map_blocks[i]
            H[i, j] = out_blocks[super][row_idx, col_idx]
            if scal2
                H[i, j] *= rt2
            end
        end
    end

    cone.hess_updated = true
    return cone.hess
end

# TODO refactor parts to make easy to implement (inv)hess prod, (inv)hess sqrt prod
# TODO need to split step 2 into two parts for sqrt prod
function H_prod_col(cone::PosSemidefTriSparse{T, R}, in_blocks) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.grad_updated
    num_super = length(cone.supers) - 1
    children = cone.children
    # parents = cone.parents
    num_cols = cone.num_cols
    num_rows = cone.num_rows
    J_rows = cone.J_rows
    L_blocks = cone.L_blocks
    V_blocks = cone.V_blocks
    inv_blocks = cone.inv_blocks

    # step 1
    for k in 1:num_super
        num_col = num_cols[k]
        num_row = num_rows[k]
        idxs_n = 1:num_col
        J = J_rows[k]
        L_block = L_blocks[k]
        in_block = in_blocks[k]

        F = zeros(R, num_row, num_row)
        F[:, idxs_n] = in_block

        for k2 in children[k]
            U2 = V_blocks[k2]
            A2 = J_rows[k2][(num_cols[k2] + 1):end]
            EJA2 = Bool[i == j for i in J, j in A2]
            F += Hermitian(EJA2 * Hermitian(U2, :L) * EJA2', :L)
        end

        if num_row > num_col
            idxs_a = (num_col + 1):num_row
            @views L_a = L_block[idxs_a, :]
            @views F_an = F[idxs_a, idxs_n]
            @views F_aa = Hermitian(F[idxs_a, idxs_a], :L)
            @views F_nn = Hermitian(F[idxs_n, idxs_n], :L)

            mul!(F_aa.data, L_a, F_an', -1, true)
            mul!(F_an, L_a, F_nn, -1, true)
            mul!(F_aa.data, F_an, L_a', -1, true)

            V_blocks[k] = F_aa
        end

        in_blocks[k] = tril!(F[:, idxs_n])
    end

    # step 2
    for k in reverse(1:num_super)
        num_col = num_cols[k]
        num_row = num_rows[k]
        idxs_n = 1:num_col
        J = J_rows[k]
        L_block = L_blocks[k]
        inv_block = inv_blocks[k]
        in_block = in_blocks[k]

        F = zeros(R, num_row, num_row)

        if num_row > num_col
            idxs_a = (num_col + 1):num_row
            V = V_blocks[k]

            @views in_block_a = in_block[idxs_a, :]
            @views F_an = mul!(F[idxs_a, idxs_n], V, in_block_a)
            copyto!(in_block_a, F_an)

            F[idxs_a, idxs_a] = V
        end

        F[:, idxs_n] = inv_block

        for k2 in children[k]
            A2 = J_rows[k2][(num_cols[k2] + 1):end]
            EJA2 = Bool[i == j for i in J, j in A2]
            V_blocks[k2] = Hermitian(EJA2' * Hermitian(F, :L) * EJA2, :L)
        end

        @views in_block_n = in_block[idxs_n, :]
        copytri!(in_block_n, 'L', cone.is_complex)
        FD_nn = LowerTriangular(L_block[idxs_n, :])
        ldiv!(FD_nn, in_block_n)
        ldiv!(FD_nn', in_block_n)
        rdiv!(in_block, FD_nn')
        rdiv!(in_block, FD_nn)
        tril!(in_block)
    end

    # step 3
    for k in reverse(1:num_super)
        num_col = num_cols[k]
        num_row = num_rows[k]
        idxs_n = 1:num_col
        J = J_rows[k]
        L_block = L_blocks[k]
        in_block = in_blocks[k]

        F = zeros(R, num_row, num_row)
        F[:, idxs_n] = in_block

        if num_row > num_col
            idxs_a = (num_col + 1):num_row
            L_a = L_block[idxs_a, :]
            @views F_an = F[idxs_a, idxs_n]
            @views F_aa = Hermitian(F[idxs_a, idxs_a], :L)
            @views F_nn = Hermitian(F[idxs_n, idxs_n], :L)

            V = V_blocks[k]
            copyto!(F_aa.data, V)

            mul!(F_nn.data, F_an', L_a, -1, true)
            mul!(F_an, F_aa, L_a, -1, true)
            mul!(F_nn.data, L_a', F_an, -1, true)
        end

        for k2 in children[k]
            A2 = J_rows[k2][(num_cols[k2] + 1):end]
            EJA2 = Bool[i == j for i in J, j in A2]
            V_blocks[k2] = Hermitian(EJA2' * Hermitian(F, :L) * EJA2, :L)
        end

        in_blocks[k] = tril!(F[:, idxs_n])
    end

    return in_blocks
end
