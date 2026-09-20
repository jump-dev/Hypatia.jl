#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

"""
$(TYPEDEF)

Linear matrix inequality cone parametrized by list of real symmetric or complex
Hermitian matrices `mats` of equal dimension.

    $(FUNCTIONNAME){T, R}(mats::Vector{AbstractMatrix{R}}, use_dual::Bool = false)
"""
mutable struct LinMatrixIneq{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    is_complex::Bool
    is_sparse::Bool
    dim::Int
    side::Int
    denseAs::Vector{Hermitian{R,Matrix{R}}}
    sparseAs::Vector{Hermitian{R,SparseMatrixCSC{R, Int}}}

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}
    use_hess_prod_slow::Bool
    use_hess_prod_slow_updated::Bool

    densesumA::Hermitian{R,Matrix{R}}
    sparsesumA::Hermitian{R,SparseMatrixCSC{R, Int}}
    densefact::Cholesky{R,Matrix{R}}
    sparsefact::SparseArrays.CHOLMOD.Factor{R, Int}
    densesumAinvAs::Vector{Hermitian{R,Matrix{R}}}
    sparsesumAinvAs::Vector{Hermitian{R,SparseMatrixCSC{R, Int}}}

    function LinMatrixIneq{T, R}(As::Vector; use_dual::Bool = false) where {T <: Real, R <: RealOrComplex{T}} 
        dim = length(As)
        @assert dim > 1
        side = size(first(As), 1)
        for A_i in As
            @assert size(A_i, 1) == side
            @assert ishermitian(A_i)
        end
        # necessary to ensure linear independence of As (but not sufficient)
        @assert svec_length(R, side) >= dim
        @assert isposdef(first(As))
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.side = side
        cone.is_complex = (R <: Complex)
        cone.is_sparse = issparse(first(As))
        if cone.is_sparse
            cone.sparseAs = Hermitian.(As)
        else
            cone.denseAs = Hermitian.(As)
        end
        return cone
    end
end

function reset_data(cone::LinMatrixIneq)
    return (
        cone.feas_updated =
            cone.grad_updated =
                cone.hess_updated =
                    cone.inv_hess_updated =
                        cone.hess_fact_updated =
                            cone.use_hess_prod_slow =
                                cone.use_hess_prod_slow_updated = false
    )
end

get_nu(cone::LinMatrixIneq) = cone.side

function set_initial_point!(arr::AbstractVector, cone::LinMatrixIneq)
    arr .= 0
    arr[1] = 1
    return arr
end

function update_feas(cone::LinMatrixIneq)
    @assert !cone.feas_updated

    if cone.is_sparse
        cone.sparsesumA = sum(wᵢ * Aᵢ for (wᵢ, Aᵢ) in zip(cone.point, cone.sparseAs))
        cone.sparsefact = cholesky(cone.sparsesumA; shift = false, check = false)
        cone.is_feas = isposdef(cone.sparsefact)
    else
        cone.densesumA = sum(wᵢ * Aᵢ for (wᵢ, Aᵢ) in zip(cone.point, cone.denseAs))
        cone.densefact = cholesky!(cone.densesumA; check = false)
        cone.is_feas = isposdef(cone.densefact)
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::LinMatrixIneq)
    @assert cone.is_feas

    if cone.is_sparse
        sL = cone.sparsefact.L
        cone.sparsesumAinvAs = [Hermitian(sL \ (sL \ A_i)', :U) for A_i in cone.sparseAs]
        @inbounds for (i, mat_i) in enumerate(cone.sparsesumAinvAs)
            cone.grad[i] = -tr(mat_i)
        end
    else
        dL = cone.densefact.L
        cone.densesumAinvAs = [Hermitian(dL \ (dL \ A_i)', :U) for A_i in cone.denseAs]
        @inbounds for (i, mat_i) in enumerate(cone.densesumAinvAs)
            cone.grad[i] = -tr(mat_i)
        end
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::LinMatrixIneq)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data

    if cone.is_sparse
        sumAinvAs = cone.sparsesumAinvAs
        @inbounds for i in 1:(cone.dim), j in i:(cone.dim)
            H[i, j] = real(dot(sumAinvAs[i], sumAinvAs[j]'))
        end
    else
        sumAinvAs = cone.densesumAinvAs
        @inbounds for i in 1:(cone.dim), j in i:(cone.dim)
            H[i, j] = real(dot(sumAinvAs[i], sumAinvAs[j]'))
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod_slow!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::LinMatrixIneq)
    cone.use_hess_prod_slow_updated || update_use_hess_prod_slow(cone)
    @assert cone.hess_updated
    cone.use_hess_prod_slow || return hess_prod!(prod, arr, cone)

    @assert cone.grad_updated
    if cone.is_sparse
        sumAinvAs = cone.sparsesumAinvAs
        @inbounds for j in 1:size(arr, 2)
            j_mat = Hermitian(sum(arr[i, j] * sumAinvAs[i] for i in 1:(cone.dim)))
            for i in 1:(cone.dim)
                prod[i, j] = real(dot(j_mat, sumAinvAs[i]))
            end
        end
    else
        sumAinvAs = cone.densesumAinvAs
        @inbounds for j in 1:size(arr, 2)
            j_mat = Hermitian(sum(arr[i, j] * sumAinvAs[i] for i in 1:(cone.dim)))
            for i in 1:(cone.dim)
                prod[i, j] = real(dot(j_mat, sumAinvAs[i]))
            end
        end
    end

    return prod
end

function dder3(cone::LinMatrixIneq, dir::AbstractVector)
    @assert cone.grad_updated
    dder3 = cone.dder3

    if cone.is_sparse
        sumAinvAs = cone.sparsesumAinvAs
        dir_mat = sum(d_i * mat_i for (d_i, mat_i) in zip(dir, sumAinvAs))
        Z = Hermitian(dir_mat * dir_mat')
        @inbounds for i in 1:(cone.dim)
            dder3[i] = real(dot(Z, sumAinvAs[i]))
        end
    else
        sumAinvAs = cone.densesumAinvAs
        dir_mat = sum(d_i * mat_i for (d_i, mat_i) in zip(dir, sumAinvAs))
        Z = Hermitian(dir_mat * dir_mat')
        @inbounds for i in 1:(cone.dim)
            dder3[i] = real(dot(Z, sumAinvAs[i]))
        end
    end

    return dder3
end

function pretty_name(cone::LinMatrixIneq)
    realorcomplex = cone.is_complex ? "complex " : "real "
    dualorprimal = use_dual_barrier(cone) ? "dual " : ""
    return realorcomplex * dualorprimal * "linear matrix inequality"
end
