#=
TODO in complex case, can maybe compute Lambda fast in feas check by taking sqrt of
point and doing outer product
=#

"""
$(TYPEDEF)

Interpolant-basis weighted sum-of-squares polynomial cone of dimension `U`, for
real or real-valued complex polynomials , parametrized by vector of matrices
`Ps` derived from interpolant basis and polynomial domain constraints.

    $(FUNCTIONNAME){T, R}(U::Int, Ps::Vector{Matrix{R}}, use_dual::Bool = false)
"""
mutable struct WSOSInterpNonnegative{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    Ps::Vector{Matrix{R}}
    nu::Int

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

    tempLL::Vector{Matrix{R}}
    tempLL2::Vector{Matrix{R}}
    tempLU::Vector{Matrix{R}}
    ΛFLP::Vector{Matrix{R}}
    tempUU::Matrix{R}
    ΛF::Vector
    Ps_times::Vector{Float64}
    Ps_order::Vector{Int}

    function WSOSInterpNonnegative{T, R}(
        U::Int,
        Ps::Vector{Matrix{R}};
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        for Pk in Ps
            @assert size(Pk, 1) == U
        end
        cone = new{T, R}()
        cone.use_dual_barrier = !use_dual # using dual barrier
        cone.dim = U
        cone.Ps = Ps
        cone.nu = sum(size(Pk, 2) for Pk in Ps)
        return cone
    end
end

reset_data(cone::WSOSInterpNonnegative) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated =
    cone.use_hess_prod_slow = cone.use_hess_prod_slow_updated = false)

function setup_extra_data!(
    cone::WSOSInterpNonnegative{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    dim = cone.dim
    Ls = [size(Pk, 2) for Pk in cone.Ps]
    cone.tempLL = [zeros(R, L, L) for L in Ls]
    cone.tempLL2 = [zeros(R, L, L) for L in Ls]
    cone.tempLU = [zeros(R, L, dim) for L in Ls]
    cone.ΛFLP = [zeros(R, L, dim) for L in Ls]
    cone.tempUU = zeros(R, dim, dim)
    K = length(Ls)
    cone.ΛF = Vector{Any}(undef, K)
    cone.Ps_times = zeros(K)
    cone.Ps_order = collect(1:K)
    return cone
end

set_initial_point!(arr::AbstractVector, cone::WSOSInterpNonnegative) = (arr .= 1)

function update_feas(cone::WSOSInterpNonnegative)
    @assert !cone.feas_updated
    D = Diagonal(cone.point)

    # order the Ps by how long it takes to check feasibility, to improve efficiency
    sortperm!(cone.Ps_order, cone.Ps_times, initialized = true) # stochastic

    cone.is_feas = true
    for k in cone.Ps_order
        cone.Ps_times[k] = @elapsed @inbounds begin
            Pk = cone.Ps[k]
            LUk = cone.tempLU[k]
            LLk = cone.tempLL[k]

            # Λ = Pk' * Diagonal(point) * Pk
            @. LUk = Pk' * cone.point' # currently faster than mul!(LUk, Pk', D)
            mul!(LLk, LUk, Pk)

            ΛFk = cone.ΛF[k] = cholesky!(Hermitian(LLk, :L), check = false)
            if !isposdef(ΛFk)
                cone.is_feas = false
                break
            end
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSInterpNonnegative)
    @assert cone.is_feas

    cone.grad .= 0
    @inbounds for k in eachindex(cone.Ps)
        ΛFLPk = cone.ΛFLP[k] # computed here
        ldiv!(ΛFLPk, cone.ΛF[k].L, cone.Ps[k]')
        @views for j in 1:cone.dim
            cone.grad[j] -= sum(abs2, ΛFLPk[:, j])
        end
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpNonnegative)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    UU = cone.tempUU

    cone.hess .= 0
    @inbounds for k in eachindex(cone.Ps)
        outer_prod!(cone.ΛFLP[k], UU, true, false)
        for j in 1:cone.dim, i in 1:j
            cone.hess.data[i, j] += abs2(UU[i, j])
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod_slow!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::WSOSInterpNonnegative,
    )
    cone.use_hess_prod_slow_updated || update_use_hess_prod_slow(cone)
    @assert cone.hess_updated
    cone.use_hess_prod_slow || return hess_prod!(prod, arr, cone)

    @assert cone.grad_updated
    prod .= 0
    @inbounds for k in eachindex(cone.Ps)
        ΛFLPk = cone.ΛFLP[k]
        LUk = cone.tempLU[k]
        LLk = cone.tempLL2[k]
        @views for j in 1:size(arr, 2)
            partial_lambda!(LUk, arr[:, j], LLk, ΛFLPk)
            for i in 1:cone.dim
                prod[i, j] += real(dot(ΛFLPk[:, i], LUk[:, i]))
            end
        end
    end
    return prod
end

function dder3(cone::WSOSInterpNonnegative, dir::AbstractVector)
    @assert cone.grad_updated
    dder3 = cone.dder3
    dder3 .= 0
    @inbounds for k in eachindex(cone.Ps)
        LUk = partial_lambda!(cone.tempLU[k], dir, cone.tempLL2[k], cone.ΛFLP[k])
        @views for j in 1:cone.dim
            dder3[j] += sum(abs2, LUk[:, j])
        end
    end
    return dder3
end

function partial_lambda!(
    LUk::Matrix,
    dir::AbstractVector,
    LLk::Matrix,
    ΛFLPk::Matrix,
    )
    mul!(LUk, ΛFLPk, Diagonal(dir))
    mul!(LLk, LUk, ΛFLPk')
    mul!(LUk, Hermitian(LLk), ΛFLPk)
    return LUk
end
