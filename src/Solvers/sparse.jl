#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

TODO play around with SuiteSparse parameters e.g. iterative refinement
TODO try not to fallback to lu
=#

abstract type SparseSystemSolver <: SystemSolver{Float64} end

release_sparse_cache(s::SparseSystemSolver) = release_sparse_cache(s.sparse_cache)

release_sparse_cache(s::SystemSolver) = nothing

function add_I_J_V(
    offset::Int,
    Is::Vector{<: Integer},
    Js::Vector{<: Integer},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    vec::Vector{Float64},
    trans::Bool,
    )
    n = length(vec)
    if !isempty(vec)
        if trans
            @views Is[offset:(offset + n - 1)] .= start_row + 1
            @views Js[offset:(offset + n - 1)] .= (start_col + 1):(start_col + n)
        else
            @views Is[offset:(offset + n - 1)] .= (start_row + 1):(start_row + n)
            @views Js[offset:(offset + n - 1)] .= start_col + 1
        end
        Vs[offset:(offset + n - 1)] .= vec
    end
    return offset + n
end

function add_I_J_V(
    offset::Int,
    Is::Vector{<: Integer},
    Js::Vector{<: Integer},
    Vs::Vector{Float64},
    start_rows::Vector{Int},
    start_cols::Vector{Int},
    vecs::Vector{Vector{Float64}},
    trans::Vector{Bool},
    )
    for (r, c, v, t) in zip(start_rows, start_cols, vecs, trans)
        offset = add_I_J_V(offset, Is, Js, Vs, r, c, v, t)
    end
    return offset
end

function add_I_J_V(
    offset::Int,
    Is::Vector{<: Integer},
    Js::Vector{<: Integer},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    mat::SparseMatrixCSC,
    trans::Bool,
    )
    for j in 1:mat.n
        col_idxs = mat.colptr[j]:(mat.colptr[j + 1] - 1)
        rows = view(mat.rowval, col_idxs)
        vals = view(mat.nzval, col_idxs)
        m = length(rows)
        if trans
            @views Is[offset:(offset + m - 1)] .= start_row + j
            @views Js[offset:(offset + m - 1)] .= start_col .+ rows
        else
            @views Is[offset:(offset + m - 1)] .= start_row .+ rows
            @views Js[offset:(offset + m - 1)] .= start_col + j
        end
        @views Vs[offset:(offset + m - 1)] .= vals
        offset += m
    end
    return offset
end

function add_I_J_V(
    offset::Int,
    Is::Vector{<: Integer},
    Js::Vector{<: Integer},
    Vs::Vector{Float64},
    start_rows::Vector{Int},
    start_cols::Vector{Int},
    mats::Vector{<: SparseMatrixCSC},
    trans::Vector{Bool},
    )
    for (r, c, m, t) in zip(start_rows, start_cols, mats, trans)
        offset = add_I_J_V(offset, Is, Js, Vs, r, c, m, t)
    end
    return offset
end

function add_I_J_V(
    offset::Int,
    Is::Vector{<: Integer},
    Js::Vector{<: Integer},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    cone::Cones.Cone,
    use_inv::Bool)
    for j in 1:Cones.dimension(cone)
        nz_rows = (use_inv ? Cones.inv_hess_nz_idxs_j(cone, j) : Cones.hess_nz_idxs_j(cone, j))
        n = length(nz_rows)
        @. Is[offset:(offset + n - 1)] = start_row + nz_rows
        @. Js[offset:(offset + n - 1)] = j + start_col
        @. Vs[offset:(offset + n - 1)] = 1
        offset += n
    end
    return offset
end
