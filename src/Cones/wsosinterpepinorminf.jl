#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

=#
using ForwardDiff

mutable struct WSOSInterpEpiNormInf{T <: Real} <: Cone{T}
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

    function WSOSInterpEpiNormInf{T}(
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
        cone = new{T}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = U * R
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache

        # soc-based
        # function barrier(point)
        #      bar = zero(eltype(point))
        #      for P in cone.Ps
        #          lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P)
        #          fact_1 = cholesky(lambda_1)
        #          for i in 2:R
        #              lambda_i = Symmetric(P' * Diagonal(point[block_idxs(U, i)]) * P)
        #              LL = fact_1.L \ lambda_i
        #              bar -= logdet(lambda_1 - LL' * LL)
        #              # bar -= logdet(lambda_1 - lambda_i * (fact_1 \ lambda_i))
        #          end
        #          bar -= logdet(fact_1)
        #      end
        #      return bar
        # end

        # orthant-based
        function barrier(point)
             bar = zero(eltype(point))
             for P in cone.Ps
                 lambda_1 = Hermitian(P' * Diagonal(point[1:U]) * P)
                 for i in 2:R
                     lambda_i = Hermitian(P' * Diagonal(point[block_idxs(U, i)]) * P)
                     bar -= logdet((lambda_1 - lambda_i) * (lambda_1 + lambda_i))
                 end
                 bar += logdet(cholesky(lambda_1)) * (R - 2)
             end
             return bar
        end
        cone.barrier = barrier

        return cone
    end
end

function setup_data(cone::WSOSInterpEpiNormInf{T}) where {T <: Real}
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

get_nu(cone::WSOSInterpEpiNormInf) = cone.R * sum(size(Pk, 2) for Pk in cone.Ps)

function set_initial_point(arr::AbstractVector, cone::WSOSInterpEpiNormInf)
    arr[1:cone.U] .= 1
    arr[(cone.U + 1):end] .= 0
    return arr
end

function update_feas(cone::WSOSInterpEpiNormInf)
    @assert !cone.feas_updated
    U = cone.U
    point = cone.point

    # cone.is_feas = true
    # @inbounds for k in eachindex(cone.Ps)
    #     P = cone.Ps[k]
    #     lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P)
    #     fact_1 = cholesky(lambda_1, check = false)
    #     if isposdef(fact_1)
    #         for i in 2:cone.R
    #             lambda_i = Symmetric(P' * Diagonal(point[block_idxs(U, i)]) * P)
    #             LL = fact_1.L \ lambda_i
    #             if !isposdef(lambda_1 - LL' * LL)
    #                 cone.is_feas = false
    #                 break
    #             end
    #         end
    #     else
    #         cone.is_feas = false
    #         break
    #     end
    # end

    cone.is_feas = true
    @inbounds for k in eachindex(cone.Ps)
        P = cone.Ps[k]
        lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P)
        fact_1 = cholesky(lambda_1, check = false)
        if isposdef(fact_1)
            for i in 2:cone.R
                lambda_i = Symmetric(P' * Diagonal(point[block_idxs(U, i)]) * P)
                if !isposdef(lambda_1 - lambda_i) || !isposdef(lambda_1 + lambda_i)
                    cone.is_feas = false
                    break
                end
            end
        else
            cone.is_feas = false
            break
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::WSOSInterpEpiNormInf) = true

function update_grad(cone::WSOSInterpEpiNormInf)
    cone.grad .= ForwardDiff.gradient(cone.barrier, cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpEpiNormInf)
    cone.hess.data .= ForwardDiff.hessian(cone.barrier, cone.point)
    cone.hess_updated = true
    return cone.hess
end

use_correction(::WSOSInterpEpiNormInf) = false
