#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

sparse lower triangle of positive semidefinite matrix cone (unscaled "smat" form)
dense(W) \in S^n : 0 >= eigmin(W)

specified with ordered lists of row and column indices for elements in lower triangle
dual cone is cone of PSD-completable matrices given the same sparsity pattern
real symmetric or complex Hermitian cases

NOTE in complex Hermitian case, on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector, but row and column indices are not repeated

see "Logarithmic barriers for sparse matrix cones" by Andersen, Dahl, Vandenberghe (2012)
but note that we do not restrict the sparsity pattern to be chordal here (at the cost of not being able to obtain "closed form" hess sqrt and inv hess oracles)
barrier is -logdet(dense(W))

NOTE only implemented for BLAS real types (Float32 and Float64) because implementation calls SuiteSparse.CHOLMOD

TODO
- improve efficiency of hessian calculations using structure
- use inbounds
- improve efficiency of frontal matrix scattering etc
- maybe allow passing more options to CHOLMOD eg through a common_struct argument to cone
=#

mutable struct PosSemidefTriSparse{T <: BlasReal, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    max_neighborhood::T
    use_heuristic_neighborhood::Bool
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
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    sparse_point
    sparse_point_map
    symb_mat
    supers
    super_map
    parents
    num_cols
    num_rows
    J_rows
    L_idxs
    F_blocks
    L_blocks
    S_blocks
    map_blocks
    temp_blocks
    rel_idxs

    function PosSemidefTriSparse{T, R}(
        side::Int,
        row_idxs::Vector{Int},
        col_idxs::Vector{Int};
        use_dual::Bool = false,
        max_neighborhood::Real = default_max_neighborhood(),
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: BlasReal}
        # check validity of inputs
        num_nz = length(row_idxs)
        @assert length(col_idxs) == num_nz
        # TODO maybe also check no off-diags appear twice?
        diag_present = falses(side)
        for (row_idx, col_idx) in zip(row_idxs, col_idxs)
            @assert col_idx <= row_idx <= side
            if row_idx == col_idx
                @assert !diag_present[row_idx] # don't count element twice
                diag_present[row_idx] = true
            end
        end
        @assert all(diag_present)
        cone = new{T, R}()
        if R <: Real
            cone.dim = num_nz
            cone.is_complex = false
        else
            cone.dim = 2 * num_nz - side
            cone.is_complex = true
        end
        @assert cone.dim >= 1
        cone.use_dual = use_dual
        cone.max_neighborhood = max_neighborhood
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.side = side # side dimension of sparse matrix
        cone.row_idxs = row_idxs
        cone.col_idxs = col_idxs
        cone.rt2 = sqrt(T(2))
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_data(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: BlasReal}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    setup_symbfact(cone)
    return
end

# setup symbolic factorization
function setup_symbfact(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: BlasReal}
    side = cone.side
    dim_R = length(cone.row_idxs)

    cm = CHOLMOD.defaults(CHOLMOD.common_struct[Base.Threads.threadid()])
    unsafe_store!(CHOLMOD.common_print[], 0)
    unsafe_store!(CHOLMOD.common_postorder[], 1)
    unsafe_store!(CHOLMOD.common_supernodal[], 2)

    fake_point = [R(l) for l in 1:dim_R]
    sparse_point = cone.sparse_point = CHOLMOD.Sparse(Hermitian(sparse(cone.row_idxs, cone.col_idxs, fake_point, side, side), :L))
    sparse_point_map = cone.sparse_point_map = zeros(Int, dim_R)
    sx_ptr = unsafe_load(pointer(sparse_point)).x
    for l in 1:dim_R
        sparse_point_map[Int(unsafe_load(sx_ptr, l))] = l
    end
    symb_mat = cone.symb_mat = CHOLMOD.fact_(sparse_point, cm)

    f = unsafe_load(pointer(symb_mat))
    @assert f.n == cone.side
    @assert f.is_ll != 0
    @assert f.is_super != 0

    num_super = Int(f.nsuper)
    supers = cone.supers = [unsafe_load(f.super, i) + 1 for i in 1:num_super]
    push!(supers, side + 1)
    fs = [unsafe_load(f.s, i) + 1 for i in 1:Int(f.ssize)]
    fpi = [unsafe_load(f.pi, i) + 1 for i in 1:num_super]
    push!(fpi, length(fs) + 1)
    fpx = [unsafe_load(f.px, i) + 1 for i in 1:num_super]

    # construct super_map
    # super_map[k] = s if column k is in supernode s
    super_map = cone.super_map = zeros(Int, side)
    for s in 1:num_super, k in supers[s]:(supers[s + 1] - 1)
        super_map[k] = s
    end

    # construct supernode tree
    parents = cone.parents = zeros(Int, num_super)
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

    num_cols = cone.num_cols = Vector{Int}(undef, num_super)
    num_rows = cone.num_rows = Vector{Int}(undef, num_super)
    L_idxs = cone.L_idxs = Vector{UnitRange{Int}}(undef, num_super)
    J_rows = cone.J_rows = Vector{Vector}(undef, num_super)
    F_blocks = cone.F_blocks = Vector{Matrix{R}}(undef, num_super)
    L_blocks = cone.L_blocks = Vector{Matrix{R}}(undef, num_super)
    S_blocks = cone.S_blocks = Vector{Matrix{R}}(undef, num_super)
    temp_blocks = cone.temp_blocks = Vector{Matrix{R}}(undef, num_super)
    rel_idxs = cone.rel_idxs = [Tuple{Int, Int}[] for k in 1:num_super]
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
        S_blocks[k] = zeros(R, num_below, num_below)
        temp_blocks[k] = zeros(R, num_row, num_col)

        if num_row > num_col
            # setup relative indices in frontal matrix
            rel_idx = rel_idxs[k]
            k_par = parents[k]
            I_k = J_k[(num_col + 1):end]
            for (idx_i, i) in enumerate(I_k), (idx_j, j) in enumerate(J_rows[k_par])
                if i == j
                    push!(rel_idx, (idx_i, idx_j))
                end
            end
        end
    end

    iperm = invperm(symb_mat.p)
    map_blocks = cone.map_blocks = Vector{Tuple{Int, Int, Int, Bool, Bool}}(undef, dim_R)
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

get_nu(cone::PosSemidefTriSparse) = cone.side

function set_initial_point(arr::AbstractVector, cone::PosSemidefTriSparse{T, T}) where {T <: BlasReal}
    for i in eachindex(arr)
        if cone.row_idxs[i] == cone.col_idxs[i]
            arr[i] = 1
        else
            arr[i] = 0
        end
    end
    return arr
end

function set_initial_point(arr::AbstractVector, cone::PosSemidefTriSparse{T, Complex{T}}) where {T <: BlasReal}
    idx = 1
    for (row_idx, col_idx) in zip(cone.row_idxs, cone.col_idxs)
        if row_idx == col_idx
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

function update_feas(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: BlasReal}
    @assert !cone.feas_updated
    point = cone.point
    sparse_point = cone.sparse_point

    # svec scale point and copy directly to CHOLDMOD Sparse data structure
    sx_ptr = unsafe_load(pointer(sparse_point)).x
    if cone.is_complex
        idx = 1
        for (p_idx, s_idx) in enumerate(cone.sparse_point_map)
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
        for (p_idx, s_idx) in enumerate(cone.sparse_point_map)
            p = point[p_idx]
            if cone.row_idxs[p_idx] != cone.col_idxs[p_idx]
                p /= cone.rt2
            end
            unsafe_store!(sx_ptr, p, s_idx)
        end
    end

    # update numeric factorization
    cone.is_feas = isposdef(CHOLMOD.cholesky!(cone.symb_mat, sparse_point; check = false))

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: BlasReal}
    @assert cone.is_feas
    num_cols = cone.num_cols
    num_rows = cone.num_rows

    # update L blocks from CHOLMOD numerical factorization
    lx_ptr = unsafe_load(pointer(cone.symb_mat)).x
    for k in 1:length(num_cols)
        L_block = cone.L_blocks[k]
        for (l, lx_idx) in enumerate(cone.L_idxs[k])
            L_block[l] = unsafe_load(lx_ptr, lx_idx)
        end
    end

    # build inv blocks
    for k in reverse(1:length(num_cols))
        num_col = num_cols[k]
        num_row = num_rows[k]
        idxs_n = 1:num_col
        F_block = cone.F_blocks[k]
        L_block = cone.L_blocks[k]

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
            F_par = Hermitian(cone.F_blocks[cone.parents[k]], :L)
            rel_idx = cone.rel_idxs[k]
            for (i, j) in rel_idx, (i2, j2) in rel_idx # TODO only lower tri
                F_aa.data[i, i2] = F_par[j, j2]
            end

            mul!(F_an, F_aa, L_a, -1, false)
            mul!(F_nn, F_an', L_a, -1, true)

            copyto!(cone.S_blocks[k], F_aa.data) # for use in Hessian calculations
        end

        @views cone.temp_blocks[k] = F_block[:, idxs_n]
    end

    smat_to_svec_sparse!(cone.grad, cone.temp_blocks, cone)
    cone.grad .*= -1

    cone.grad_updated = true
    return cone.grad
end

# for each column in identity, get hess prod to build explicit hess
function update_hess(cone::PosSemidefTriSparse{T, R}) where {R <: RealOrComplex{T}} where {T <: BlasReal}
    @assert cone.grad_updated
    temp_blocks = cone.temp_blocks
    rt2 = cone.rt2
    invrt2 = inv(rt2)
    H = cone.hess.data

    H_idx_j = 1
    for (j, (super_j, row_idx_j, col_idx_j, scal_j, swapped_j)) in enumerate(cone.map_blocks)
        for H_block in temp_blocks
            H_block .= 0
        end

        if cone.is_complex
            # real part j
            temp_blocks[super_j][row_idx_j, col_idx_j] = (scal_j ? invrt2 : 1)
            out_blocks = _hess_prod_blocks(cone, temp_blocks)
            H_idx_i = 1
            for i in 1:j
                (super_i, row_idx_i, col_idx_i, scal_i, swapped_i) = cone.map_blocks[i]
                HijR = out_blocks[super_i][row_idx_i, col_idx_i]
                if scal_i
                    HijR *= rt2
                end
                H[H_idx_i, H_idx_j] = real(HijR) # real part i
                H_idx_i += 1
                if row_idx_i != col_idx_i
                    H[H_idx_i, H_idx_j] = swapped_i ? -imag(HijR) : imag(HijR) # complex part i
                    H_idx_i += 1
                end
            end
            H_idx_j += 1

            if row_idx_j != col_idx_j
                # complex part j
                for H_block in temp_blocks
                    H_block .= 0
                end
                temp_blocks[super_j][row_idx_j, col_idx_j] = (scal_j ? invrt2 : 1) * im
                out_blocks = _hess_prod_blocks(cone, temp_blocks)
                H_idx_i = 1
                for i in 1:j
                    (super_i, row_idx_i, col_idx_i, scal_i, swapped_i) = cone.map_blocks[i]
                    HijR = out_blocks[super_i][row_idx_i, col_idx_i]
                    if scal_i
                        HijR *= rt2
                    end
                    H[H_idx_i, H_idx_j] = swapped_j ? -real(HijR) : real(HijR) # real part i
                    H_idx_i += 1
                    if row_idx_i != col_idx_i
                        H[H_idx_i, H_idx_j] = xor(swapped_i, swapped_j) ? -imag(HijR) : imag(HijR) # complex part i
                        H_idx_i += 1
                    end
                end
                H_idx_j += 1
            end
        else
            temp_blocks[super_j][row_idx_j, col_idx_j] = (scal_j ? invrt2 : 1)
            out_blocks = _hess_prod_blocks(cone, temp_blocks)
            for i in 1:j
                (super_i, row_idx_i, col_idx_i, scal_i, swapped_i) = cone.map_blocks[i]
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

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::PosSemidefTriSparse)
    @assert cone.grad_updated
    temp_blocks = cone.temp_blocks
    @inbounds for i in 1:size(arr, 2)
        @views svec_to_smat_sparse!(temp_blocks, arr[:, i], cone)
        _hess_prod_blocks(cone, temp_blocks)
        @views smat_to_svec_sparse!(prod[:, i], temp_blocks, cone)
    end
    return prod
end

function _hess_prod_blocks(cone::PosSemidefTriSparse{T, R}, temp_blocks) where {R <: RealOrComplex{T}} where {T <: BlasReal}
    _hess_step1(cone, temp_blocks)
    _hess_step2(cone, temp_blocks)
    _hess_step3(cone, temp_blocks)
    return temp_blocks
end

function _hess_step1(cone::PosSemidefTriSparse{T, R}, temp_blocks) where {R <: RealOrComplex{T}} where {T <: BlasReal}
    @assert cone.grad_updated

    for k in eachindex(cone.num_cols)
        num_col = cone.num_cols[k]
        num_row = cone.num_rows[k]
        idxs_n = 1:num_col
        idxs_a = (num_col + 1):num_row
        F_block = cone.F_blocks[k]
        temp_block = temp_blocks[k]

        F_block[idxs_a, idxs_a] .= 0
        F_block[:, idxs_n] = temp_block
    end

    for k in eachindex(cone.num_cols)
        num_col = cone.num_cols[k]
        num_row = cone.num_rows[k]
        idxs_n = 1:num_col
        F_block = cone.F_blocks[k]
        temp_block = temp_blocks[k]

        if num_row > num_col
            idxs_a = (num_col + 1):num_row
            @views L_a = cone.L_blocks[k][idxs_a, :]
            @views F_an = F_block[idxs_a, idxs_n]
            @views F_aa = Hermitian(F_block[idxs_a, idxs_a], :L)
            @views F_nn = Hermitian(F_block[idxs_n, idxs_n], :L)

            mul!(F_aa.data, L_a, F_an', -1, true)
            mul!(F_an, L_a, F_nn, -1, true)
            mul!(F_aa.data, F_an, L_a', -1, true)

            F_par = cone.F_blocks[cone.parents[k]]
            rel_idx = cone.rel_idxs[k]
            for (i, j) in rel_idx, (i2, j2) in rel_idx # TODO only lower tri
                F_par[j, j2] += F_aa[i, i2]
            end
        end

        @views copyto!(temp_block, F_block[:, idxs_n])
    end

    return temp_blocks
end

function _hess_step2(cone::PosSemidefTriSparse{T, R}, temp_blocks) where {R <: RealOrComplex{T}} where {T <: BlasReal}
    for k in 1:length(cone.num_cols)
        num_col = cone.num_cols[k]
        num_row = cone.num_rows[k]
        idxs_n = 1:num_col
        temp_block = temp_blocks[k]

        if num_row > num_col
            idxs_a = (num_col + 1):num_row
            @views temp_block_a = temp_block[idxs_a, :]
            @views F_an = cone.F_blocks[k][idxs_a, idxs_n]
            mul!(F_an, Hermitian(cone.S_blocks[k], :L), temp_block_a)
            copyto!(temp_block_a, F_an)
        end

        @views temp_block_n = temp_block[idxs_n, :]
        copytri!(temp_block_n, 'L', cone.is_complex)
        @views L_n = LowerTriangular(cone.L_blocks[k][idxs_n, :])
        ldiv!(L_n, temp_block_n)
        ldiv!(L_n', temp_block_n)
        rdiv!(temp_block, L_n')
        rdiv!(temp_block, L_n)
    end

    return temp_blocks
end

function _hess_step3(cone::PosSemidefTriSparse{T, R}, temp_blocks) where {R <: RealOrComplex{T}} where {T <: BlasReal}
    for k in reverse(1:length(cone.num_cols))
        num_col = cone.num_cols[k]
        num_row = cone.num_rows[k]
        idxs_n = 1:num_col
        F_block = cone.F_blocks[k]
        temp_block = temp_blocks[k]

        F_block[:, idxs_n] = temp_block

        if num_row > num_col
            idxs_a = (num_col + 1):num_row
            @views L_a = cone.L_blocks[k][idxs_a, :]
            @views F_an = F_block[idxs_a, idxs_n]
            @views F_aa = Hermitian(F_block[idxs_a, idxs_a], :L)
            @views F_nn = Hermitian(F_block[idxs_n, idxs_n], :L)

            F_aa.data .= 0
            F_par = Hermitian(cone.F_blocks[cone.parents[k]], :L)
            rel_idx = cone.rel_idxs[k]
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

function svec_to_smat_sparse!(blocks::Vector{Matrix{T}}, vec::AbstractVector{T}, cone::PosSemidefTriSparse{T, T}) where {T <: BlasReal}
    for b in blocks
        b .= 0
    end
    for (i, (super, row_idx, col_idx, scal, swapped)) in enumerate(cone.map_blocks)
        vec_i = vec[i]
        if scal
            vec_i /= cone.rt2
        end
        blocks[super][row_idx, col_idx] = vec_i
    end
    return blocks
end

function smat_to_svec_sparse!(vec::AbstractVector{T}, blocks::Vector{Matrix{T}}, cone::PosSemidefTriSparse{T, T}) where {T <: BlasReal}
    for (i, (super, row_idx, col_idx, scal, swapped)) in enumerate(cone.map_blocks)
        vec_i = blocks[super][row_idx, col_idx]
        if scal
            vec_i *= cone.rt2
        end
        vec[i] = vec_i
    end
    return vec
end

function svec_to_smat_sparse!(blocks::Vector{Matrix{Complex{T}}}, vec::AbstractVector{T}, cone::PosSemidefTriSparse{T, Complex{T}}) where {T <: BlasReal}
    for b in blocks
        b .= 0
    end
    idx = 1
    for (i, (super, row_idx, col_idx, scal, swapped)) in enumerate(cone.map_blocks)
        if scal
            blocks[super][row_idx, col_idx] = Complex(vec[idx], (swapped ? -vec[idx + 1] : vec[idx + 1])) / cone.rt2
            idx += 2
        else
            blocks[super][row_idx, col_idx] = vec[idx]
            idx += 1
        end
    end
    return blocks
end

function smat_to_svec_sparse!(vec::AbstractVector{T}, blocks::Vector{Matrix{Complex{T}}}, cone::PosSemidefTriSparse{T, Complex{T}}) where {T <: BlasReal}
    idx = 1
    for (i, (super, row_idx, col_idx, scal, swapped)) in enumerate(cone.map_blocks)
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
    return vec
end
