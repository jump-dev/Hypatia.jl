#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of L-infinity norm
(u in R, w in R^n) : u >= norm_inf(w)

barrier from "Barrier Functions in Interior Point Methods" by Osman Guler
-sum_i(log(u - w_i^2/u)) - log(u)
= -sum_i(log(u^2 - w_i^2)) + (n - 1)log(u)
=#

mutable struct EpiNormInf{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    max_neighborhood::T
    dim::Int
    n::Int
    is_complex::Bool
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_aux_updated::Bool
    hess_sqrt_aux_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, SparseMatrixCSC{T, Int}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_sqrt::UpperTriangular{T, SparseMatrixCSC{T, Int}}
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    w::AbstractVector{R}
    den::AbstractVector{T}
    uden::Vector{R}
    wden::Vector{R}
    Huu::T
    Hure::Vector{T}
    Huim::Vector{T}
    Hrere::Vector{T}
    Hreim::Vector{T}
    Himim::Vector{T}
    schur::T

    function EpiNormInf{T, R}(
        dim::Int;
        use_dual::Bool = false,
        max_neighborhood::Real = default_max_neighborhood(),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.is_complex = (R <: Complex)
        cone.n = (cone.is_complex ? div(dim - 1, 2) : dim - 1)
        return cone
    end
end

reset_data(cone::EpiNormInf) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated = cone.hess_sqrt_aux_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    n = cone.n
    cone.w = zeros(R, n)
    cone.den = zeros(T, n)
    cone.wden = zeros(R, n)
    cone.uden = zeros(R, n)
    cone.Hure = zeros(T, n)
    cone.Hrere = zeros(T, n)
    if cone.is_complex
        cone.Huim = zeros(T, n)
        cone.Hreim = zeros(T, n)
        cone.Himim = zeros(T, n)
    end
    return
end

use_correction(cone::EpiNormInf) = true

get_nu(cone::EpiNormInf) = cone.n + 1

function set_initial_point(arr::AbstractVector{T}, cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormInf{T}) where {T}
    @assert !cone.feas_updated
    u = cone.point[1]
    @views vec_copy_to!(cone.w, cone.point[2:end])
    cone.is_feas = (u > eps(T) && u - norm(cone.w, Inf) > eps(T))
    cone.feas_updated = true
    return cone.is_feas
end

function update_dual_feas(cone::EpiNormInf{T}) where {T}
    dp = cone.dual_point
    u = dp[1]
    if u > eps(T)
        if cone.is_complex
            norm1 = sum(hypot(dp[2i], dp[2i + 1]) for i in 1:cone.n)
        else
            @views norm1 = norm(dp[2:end], 1)
        end
        return (u - norm1 > eps(T))
    end
    return false
end

function update_grad(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = cone.w

    usqr = abs2(u)
    minval = eps(T)
    @. cone.den = max(T(0.5) * (usqr - abs2(w)), minval)
    @. cone.uden = u / cone.den
    @. cone.wden = w / cone.den
    cone.grad[1] = (length(w) - 1) / u - sum(cone.uden)
    @views vec_copy_to!(cone.grad[2:end], cone.wden)

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.grad_updated
    u = cone.point[1]
    w = cone.w

    sumiden = zero(T)
    usqr = abs2(u)
    @inbounds for (j, wj) in enumerate(w)
        wdenj = cone.wden[j]
        Huj = wdenj * -cone.uden[j]
        invdenj = inv(cone.den[j])
        if cone.is_complex
            (cone.Hure[j], cone.Huim[j]) = reim(Huj)
            cone.Hrere[j] = abs2(real(wdenj)) + invdenj
            cone.Himim[j] = abs2(imag(wdenj)) + invdenj
            cone.Hreim[j] = real(wdenj) * imag(wdenj)
        else
            cone.Hure[j] = Huj
            cone.Hrere[j] = abs2(wdenj) + invdenj
        end
        sumiden += invdenj
    end

    t1 = (cone.n - 1) / u / u
    cone.Huu = sum(abs2, cone.uden) - t1 - sumiden
    @assert cone.Huu > 0
    cone.schur = sumiden - t1
    @assert cone.schur > 0

    cone.hess_aux_updated = true
    return
end

function update_hess(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)

    if !isdefined(cone, :hess)
        # initialize sparse idxs for upper triangle of Hessian
        spfun = (cone.is_complex ? sparse_upper_arrow_block2 : sparse_upper_arrow)
        cone.hess = Symmetric(spfun(T, cone.n), :U)
    end

    # modify nonzeros of upper triangle of Hessian
    nzval = cone.hess.data.nzval
    nzval[1] = cone.Huu
    if cone.is_complex
        nz_idx = 1
        @inbounds for i in 1:cone.n
            @. nzval[nz_idx .+ (1:5)] = (cone.Hure[i], cone.Hrere[i], cone.Huim[i], cone.Hreim[i], cone.Himim[i])
            nz_idx += 5
        end
    else
        nz_idx = 2
        @inbounds for i in 1:cone.n
            nzval[nz_idx] = cone.Hure[i]
            nzval[nz_idx + 1] = cone.Hrere[i]
            nz_idx += 2
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiNormInf{T}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)
    if !isdefined(cone, :inv_hess)
        cone.inv_hess = Symmetric(zeros(T, cone.dim, cone.dim), :U)
    end
    Hi = cone.inv_hess.data
    wden = cone.wden
    u = cone.point[1]
    minval = eps(T)

    Hi[1, 1] = 1
    @views Hiu = Hi[1, 2:end]
    vec_copy_to!(Hiu, wden)
    Hiu .*= -u
    @inbounds for j in 2:cone.dim, i in 2:j
        Hi[i, j] = Hi[1, j] * Hi[1, i]
    end
    Hi ./= max(cone.schur, minval)
    if cone.is_complex
        @inbounds for j in 1:cone.n
            rerej = cone.Hrere[j]
            reimj = cone.Hreim[j]
            imimj = cone.Himim[j]
            detj = max(rerej * imimj - abs2(reimj), minval)
            vj = 2j
            wj = vj + 1
            Hi[vj, vj] += imimj / detj
            Hi[wj, wj] += rerej / detj
            Hi[vj, wj] -= reimj / detj
        end
    else
        @inbounds for (j, rerej) in enumerate(cone.Hrere)
            vj = j + 1
            Hi[vj, vj] += inv(max(rerej, minval))
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

# auxiliary calculations for sqrt prod and hess prod oracles
function update_hess_sqrt_aux(cone::EpiNormInf{T}) where {T}
    cone.hess_aux_updated || update_hess_aux(cone)
    @assert !cone.hess_sqrt_aux_updated

    if !isdefined(cone, :hess_sqrt)
        # initialize sparse idxs for upper triangular factor of Hessian
        spfun = (cone.is_complex ? sparse_upper_arrow_block2 : sparse_upper_arrow)
        cone.hess_sqrt = UpperTriangular(spfun(T, cone.n))
    end

    # modify nonzeros of upper triangular factor of inverse Hessian
    if cone.is_complex
        factor_upper_arrow_block2(cone.Huu, cone.Hure, cone.Huim, cone.Hrere, cone.Hreim, cone.Himim, cone.hess_sqrt.data.nzval)
    else
        factor_upper_arrow(cone.Huu, cone.Hure, cone.Hrere, cone.hess_sqrt.data.nzval)
    end

    cone.hess_sqrt_aux_updated = true
    return
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    cone.hess_sqrt_aux_updated || update_hess_sqrt_aux(cone)
    copyto!(prod, arr)
    lmul!(cone.hess_sqrt, lmul!(cone.hess_sqrt', prod))
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    cone.hess_sqrt_aux_updated || update_hess_sqrt_aux(cone)
    ldiv!(cone.hess_sqrt', ldiv!(prod, cone.hess_sqrt, arr))
    return prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    cone.hess_sqrt_aux_updated || update_hess_sqrt_aux(cone)
    copyto!(prod, arr)
    lmul!(cone.hess_sqrt', prod)
    return prod
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    cone.hess_sqrt_aux_updated || update_hess_sqrt_aux(cone)
    ldiv!(prod, cone.hess_sqrt, arr)
    return prod
end

function correction2(cone::EpiNormInf{T, R}, primal_dir::AbstractVector{T}) where {R <: RealOrComplex{T}} where {T <: Real}
    u = cone.point[1]
    udir = primal_dir[1]
    corr = cone.correction

    corr1 = T(-0.5) * u * udir * sum((3 - 2 * u / z * u) / z / z for z in cone.den) * udir - (cone.n - 1) * abs2(udir / u) / u
    for i in 1:cone.n
        deni = -4 * cone.den[i]
        udeni = 2 * cone.uden[i]
        suuw = udir * (-1 + udeni * u)
        wi = cone.w[i]
        wdeni = 2 * cone.wden[i]
        if cone.is_complex
            (wdenire, wdeniim) = reim(wdeni)
            (wire, wiim) = reim(wi)
            (dire, diim) = (primal_dir[2i], primal_dir[2i + 1])
            uuwre = suuw * wdenire
            uuwim = suuw * wdeniim
            uimimre = 1 + wdenire * wire
            uimimim = 1 + wdeniim * wiim
            uimimrere = -udeni * uimimre * dire
            uimimimim = -udeni * uimimim * diim
            uimimimre = -udeni * wdeniim * wire
            imimwrerere = wdenire * (2 + uimimre)
            imimwimimim = wdeniim * (2 + uimimim)
            imimwrereim = wdeniim * uimimre * dire
            imimwimimre = wdenire * uimimim * diim
            corr1 += (2 * (uuwre * dire + uuwim * diim) + uimimrere * dire + uimimimim * diim + 2 * uimimimre * diim * dire) / deni
            corr[2i] = (udir * (2 * (uimimrere + uimimimre * diim) + uuwre) + (abs2(dire) * imimwrerere + diim * (2 * imimwrereim + imimwimimre))) / deni
            corr[2i + 1] = (udir * (2 * (uimimimim + uimimimre * dire) + uuwim) + (abs2(diim) * imimwimimim + dire * (2 * imimwimimre + imimwrereim))) / deni
        else
            di = primal_dir[1 + i]
            uuw = suuw * wdeni
            uimim = 1 + wdeni * wi
            uimim2 = -udeni * uimim * di
            corr1 += di * (2 * uuw + uimim2) / deni
            corr[1 + i] = (udir * (uuw + 2 * uimim2) + di * wdeni * (2 + uimim) * di) / deni
        end
    end
    corr[1] = corr1

    return corr
end

hess_nz_count(cone::EpiNormInf{<:Real, <:Real}) = 3 * cone.dim - 2
hess_nz_count(cone::EpiNormInf{<:Real, <:Complex}) = 3 * cone.dim - 2 + 2 * cone.n
hess_nz_count_tril(cone::EpiNormInf{<:Real, <:Real}) = 2 * cone.dim - 1
hess_nz_count_tril(cone::EpiNormInf{<:Real, <:Complex}) = 2 * cone.dim - 1 + cone.n
hess_nz_idxs_col(cone::EpiNormInf{<:Real, <:Real}, j::Int) = (j == 1 ? (1:cone.dim) : [1, j])
hess_nz_idxs_col(cone::EpiNormInf{<:Real, <:Complex}, j::Int) = (j == 1 ? (1:cone.dim) : (iseven(j) ? [1, j, j + 1] : [1, j - 1, j]))
hess_nz_idxs_col_tril(cone::EpiNormInf{<:Real, <:Real}, j::Int) = (j == 1 ? (1:cone.dim) : [j])
hess_nz_idxs_col_tril(cone::EpiNormInf{<:Real, <:Complex}, j::Int) = (j == 1 ? (1:cone.dim) : (iseven(j) ? [j, j + 1] : [j]))
