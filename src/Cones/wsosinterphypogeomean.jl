#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

=#
using ForwardDiff
using GenericLinearAlgebra

mutable struct WSOSInterpHypoGeoMean{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    point::AbstractVector{T}
    dual_point::AbstractVector{T}
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
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    barrier::Function

    function WSOSInterpHypoGeoMean{T}(
        R::Int,
        U::Int,
        Ps::Vector{Matrix{T}};
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        for Pk in Ps
            @assert size(Pk, 1) == U
        end
        @assert R == 3
        cone = new{T}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = U * R
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache

        function barrier(point)
             bar = zero(eltype(point))
             for P in cone.Ps
                 lambda_1 = Hermitian(P' * Diagonal(point[1:U]) * P)
                 fact_1 = cholesky(-lambda_1)

                 lambda_2 = Hermitian(P' * Diagonal(point[block_idxs(U, 2)]) * P) * 2
                 (vals_2, vecs_2) = GenericLinearAlgebra.eigen(lambda_2)
                 sqrt_vals_2 = sqrt.(vals_2)
                 sqrt_2 = vecs_2 * Diagonal(sqrt_vals_2) * vecs_2'
                 sqrt_2_i =  vecs_2 * Diagonal(inv.(sqrt_vals_2)) * vecs_2'
                 logdet_2 = sum(log.(vals_2))

                 lambda_3 = Hermitian(P' * Diagonal(point[block_idxs(U, 3)]) * P) * 2

                 inner = Hermitian(sqrt_2_i * lambda_3 * sqrt_2_i)
                 (vals_inner, vecs_inner) = GenericLinearAlgebra.eigen(inner)
                 half_geomean = sqrt_2 * vecs_inner * Diagonal(sqrt.(sqrt.(vals_inner)))
                 # sqrt_inner = vecs_inner * Diagonal(sqrt.(vals_inner)) * vecs_inner'
                 # geomean = sqrt_2 * sqrt_inner * sqrt_2
                 geomean = Hermitian(half_geomean * half_geomean')

                 # @show vals_2
                 # @show GenericLinearAlgebra.eigvals(lambda_3)
                 # @show GenericLinearAlgebra.eigvals(-lambda_1)
                 # @show GenericLinearAlgebra.eigvals(geomean + lambda_1)

                 bar -= logdet(geomean + lambda_1) + logdet(fact_1) + (logdet(cholesky(lambda_3)) + logdet_2) / 2
             end
             return bar
        end
        cone.barrier = barrier

        return cone
    end
end

function setup_data(cone::WSOSInterpHypoGeoMean{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = similar(cone.point)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    return
end

get_nu(cone::WSOSInterpHypoGeoMean) = cone.R * sum(size(Pk, 2) for Pk in cone.Ps)

function set_initial_point(arr::AbstractVector, cone::WSOSInterpHypoGeoMean)
    arr[1:cone.U] .= -1
    arr[(cone.U + 1):end] .= 2
    @show arr, size(cone.point)
    return arr
end

function update_feas(cone::WSOSInterpHypoGeoMean)
    @assert !cone.feas_updated
    U = cone.U
    point = cone.point

    cone.is_feas = true
    @inbounds for k in eachindex(cone.Ps)
        P = cone.Ps[k]
        lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P)

        lambda_2 = Hermitian(P' * Diagonal(point[block_idxs(U, 2)]) * P) * 2
        (vals_2, vecs_2) = fact_2 = GenericLinearAlgebra.eigen(lambda_2)
        sqrt_vals_2 = sqrt.(vals_2)
        sqrt_2 = vecs_2 * Diagonal(sqrt_vals_2) * vecs_2'
        sqrt_2_i =  vecs_2 * Diagonal(inv.(sqrt_vals_2)) * vecs_2'

        lambda_3 = Symmetric(P' * Diagonal(point[block_idxs(U, 3)]) * P) * 2

        if !isposdef(fact_2) || !isposdef(lambda_3) || !isposdef(-lambda_1)
            cone.is_feas = false
            break
        end

        geomean = Hermitian(sqrt_2 * sqrt(sqrt_2_i * lambda_3 * sqrt_2_i) * sqrt_2)

        @show vals_2
        @show GenericLinearAlgebra.eigvals(lambda_3)
        @show GenericLinearAlgebra.eigvals(-lambda_1)
        @show GenericLinearAlgebra.eigvals(geomean + lambda_1)

        if !isposdef(geomean + lambda_1)
            cone.is_feas = false
            break
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::WSOSInterpHypoGeoMean) = true

function update_grad(cone::WSOSInterpHypoGeoMean)
    cone.grad .= ForwardDiff.gradient(cone.barrier, cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpHypoGeoMean)
    cone.hess.data .= ForwardDiff.hessian(cone.barrier, cone.point)
    cone.hess_updated = true
    @show eigvals(cone.hess)
    return cone.hess
end

use_correction(::WSOSInterpHypoGeoMean) = false
