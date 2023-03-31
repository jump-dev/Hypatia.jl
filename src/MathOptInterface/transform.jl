#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

# transformations fallbacks
untransform_affine(::MOI.AbstractVectorSet, vals::AbstractVector) = vals
needs_untransform(::MOI.AbstractVectorSet) = false
needs_rescale(::MOI.AbstractVectorSet) = false
needs_permute(::MOI.AbstractVectorSet) = false

# transformations (transposition of matrix) for MOI rectangular matrix
# cones with matrix of more rows than columns
const SpecNucCone = Union{MOI.NormSpectralCone, MOI.NormNuclearCone}

needs_untransform(cone::SpecNucCone) = (cone.row_dim > cone.column_dim)

function untransform_affine(cone::SpecNucCone, vals::AbstractVector)
    if needs_untransform(cone)
        @views vals[2:end] = reshape(vals[2:end], cone.column_dim, cone.row_dim)'
    end
    return vals
end

needs_permute(cone::SpecNucCone) = needs_untransform(cone)

function permute_affine(cone::SpecNucCone, vals::AbstractVector{T}) where {T}
    w_vals = reshape(vals[2:end], cone.row_dim, cone.column_dim)'
    return vcat(vals[1], vec(w_vals))
end

function permute_affine(cone::SpecNucCone, func::VAF{T}) where {T}
    terms = func.terms
    idxs_new = zeros(Int, length(terms))
    for k in eachindex(idxs_new)
        i = terms[k].output_index
        @assert i >= 1
        if i <= 2
            idxs_new[k] = i
            continue
        end
        (col_old, row_old) = divrem(i - 2, cone.row_dim)
        k_idx = row_old * cone.column_dim + col_old + 2
        idxs_new[k] = terms[k_idx].output_index
    end
    return idxs_new
end

# transformations (svec rescaling) for MOI symmetric matrix cones not
# in svec (scaled lower triangle) form
const SvecCone = Union{
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.LogDetConeTriangle,
    MOI.RootDetConeTriangle,
}

svec_offset(::MOI.PositiveSemidefiniteConeTriangle) = 1
svec_offset(::MOI.RootDetConeTriangle) = 2
svec_offset(::MOI.LogDetConeTriangle) = 3

needs_untransform(::SvecCone) = true

function untransform_affine(cone::SvecCone, vals::AbstractVector{T}) where {T}
    @views svec_vals = vals[svec_offset(cone):end]
    Cones.scale_svec!(svec_vals, inv(sqrt(T(2))))
    return vals
end

needs_rescale(::SvecCone) = true

function rescale_affine(cone::SvecCone, vals::AbstractVector{T}) where {T}
    @views svec_vals = vals[svec_offset(cone):end]
    Cones.scale_svec!(svec_vals, sqrt(T(2)))
    return vals
end

function rescale_affine(cone::SvecCone, func::VAF{T}, vals::AbstractVector{T}) where {T}
    scal_start = svec_offset(cone) - 1
    rt2 = sqrt(T(2))
    for i in eachindex(vals)
        k = func.terms[i].output_index - scal_start
        if k > 0 && !MOI.Utilities.is_diagonal_vectorized_index(k)
            vals[i] *= rt2
        end
    end
    return vals
end

# transformation (svec rescaling and real/imag parts reordering) for MOI Hermitian PSD cone not in svec (scaled lower triangle) form
needs_untransform(::MOI.HermitianPositiveSemidefiniteConeTriangle) = true

function untransform_affine(
    cone::MOI.HermitianPositiveSemidefiniteConeTriangle,
    vals::AbstractVector{T},
) where {T}
    irt2 = inv(sqrt(T(2)))
    side = isqrt(MOI.dimension(cone))
    k_re = 1
    k_im = Cones.svec_length(side) + 1
    l = 1
    new_vals = zero(vals)
    for i in 1:side
        for j in 1:(i - 1)
            new_vals[k_re] = vals[l] * irt2
            new_vals[k_im] = vals[l + 1] * irt2
            k_re += 1
            k_im += 1
            l += 2
        end
        new_vals[k_re] = vals[l]
        k_re += 1
        l += 1
    end
    @assert l == 1 + length(vals)
    return new_vals
end

needs_rescale(::MOI.HermitianPositiveSemidefiniteConeTriangle) = true

function rescale_affine(
    cone::MOI.HermitianPositiveSemidefiniteConeTriangle,
    vals::AbstractVector{T},
) where {T}
    rt2 = sqrt(T(2))
    side = isqrt(MOI.dimension(cone))
    k_re = 1
    k_im = Cones.svec_length(side) + 1
    for i in 1:side
        for j in 1:(i - 1)
            vals[k_re] *= rt2
            k_re += 1
            vals[k_im] *= rt2
            k_im += 1
        end
        k_re += 1
    end
    return vals
end

# function rescale_affine(
#     cone::MOI.HermitianPositiveSemidefiniteConeTriangle,
#     func::VAF{T},
#     vals::AbstractVector{T},
# ) where {T}
#     rt2 = sqrt(T(2))
#     for i in eachindex(vals)
#         k = func.terms[i].output_index
#         if k > 0 && isqrt(k)^2 != k
#             vals[i] *= rt2
#         end
#     end
#     return vals
# end

needs_permute(cone::MOI.HermitianPositiveSemidefiniteConeTriangle) = true

function permute_affine(
    cone::MOI.HermitianPositiveSemidefiniteConeTriangle,
    vals::AbstractVector{T},
) where {T}
    side = isqrt(MOI.dimension(cone))
    k_re = 1
    k_im = Cones.svec_length(side) + 1
    l = 1
    new_vals = zero(vals)
    for i in 1:side
        for j in 1:(i - 1)
            new_vals[k_re] = vals[l]
            new_vals[k_im] = vals[l + 1]
            k_re += 1
            k_im += 1
            l += 2
        end
        new_vals[k_re] = vals[l]
        k_re += 1
        l += 1
    end
    @assert l == 1 + length(vals)
    return new_vals
end

function vec_to_symm_idxs(k::Int)
    i = div(1 + isqrt(8 * k - 7), 2)
    j = k - div((i - 1) * i, 2)
    @assert i <= j
    return (i, j)
end

# function permute_affine(
#     cone::MOI.HermitianPositiveSemidefiniteConeTriangle,
#     func::VAF{T},
# ) where {T}
#     side = isqrt(MOI.dimension(cone))
#     re_len = Cones.svec_length(side)
#     terms = func.terms
#     idxs_new = zeros(Int, length(terms))
#     for k in eachindex(idxs_new)
#         i = terms[k].output_index
#         @assert i >= 1
#         l = terms[k_idx].output_index
#         if l > re_len
#             # imag
#             pos = l - re_len
#             (i1, j1) = vec_to_symm_idxs(pos)
#             idxs_new[k] = j1^2 + 2 * i1
#         else
#             # real
#             (i1, j1) = vec_to_symm_idxs(l)
#             idxs_new[k] = (j1 - 1)^2 + 2 * i1 - 1
#         end
#     end
#     return idxs_new
# end
