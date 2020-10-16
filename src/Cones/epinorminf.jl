#=
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
    grad::Vector{T}
    correction::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_aux_updated::Bool
    hess_sqrt_aux_updated::Bool
    inv_hess_aux_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, SparseMatrixCSC{T, Int}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_sqrt::UpperTriangular{T, SparseMatrixCSC{T, Int}}
    no_sqrts::Bool

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
    Hiure::Vector{T}
    Hiuim::Vector{T}
    schur::T
    idet::Vector{T}

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

use_heuristic_neighborhood(cone::EpiNormInf) = false

reset_data(cone::EpiNormInf) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated = cone.hess_sqrt_aux_updated = cone.inv_hess_aux_updated = false)

function use_sqrt_oracles(cone::EpiNormInf)
    cone.no_sqrts && return false
    cone.hess_sqrt_aux_updated || update_hess_sqrt_aux(cone)
    return !cone.no_sqrts
end

# TODO only allocate the fields we use
function setup_extra_data(cone::EpiNormInf{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    n = cone.n
    cone.w = zeros(R, n)
    cone.wden = zeros(R, n)
    cone.den = zeros(T, n)
    cone.uden = zeros(R, n)
    cone.Hure = zeros(T, n)
    cone.Hrere = zeros(T, n)
    cone.Hiure = zeros(T, n)
    if cone.is_complex
        cone.Huim = zeros(T, n)
        cone.Hreim = zeros(T, n)
        cone.Himim = zeros(T, n)
        cone.Hiuim = zeros(T, n)
        cone.idet = zeros(T, n)
    end
    return cone
end

get_nu(cone::EpiNormInf) = cone.n + 1

function set_initial_point(arr::AbstractVector{T}, cone::EpiNormInf{T}) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormInf{T}) where {T}
    @assert !cone.feas_updated
    u = cone.point[1]
    @views vec_copy_to!(cone.w, cone.point[2:end])

    cone.is_feas = (u > eps(T) && abs2(u) - maximum(abs2, cone.w) > eps(T))

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiNormInf{T}) where {T}
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
    den = cone.den

    for (j, wj) in enumerate(w)
        absw = abs(wj)
        den[j] = T(0.5) * (u - absw) * (u + absw)
    end
    @. cone.uden = u / den
    @. cone.wden = w / den
    cone.grad[1] = (cone.n - 1) / u - sum(cone.uden)
    @views vec_copy_to!(cone.grad[2:end], cone.wden)

    cone.grad_updated = true
    return cone.grad
end

function update_hess_aux(cone::EpiNormInf{T}) where {T <: Real}
    @assert cone.grad_updated
    u = cone.point[1]
    w = cone.w
    uden = cone.uden

    @inbounds for (j, wdenj) in enumerate(cone.wden)
        udenj = uden[j]
        invdenj = inv(cone.den[j])
        if cone.is_complex
            (wdre, wdim) = reim(wdenj)
            cone.Hure[j] = -wdre * udenj
            cone.Huim[j] = -wdim * udenj
            cone.Hrere[j] = abs2(wdre) + invdenj
            cone.Himim[j] = abs2(wdim) + invdenj
            cone.Hreim[j] = wdre * wdim
        else
            cone.Hure[j] = -wdenj * udenj
            cone.Hrere[j] = abs2(wdenj) + invdenj
        end
    end
    cone.Huu = sum(abs2, uden) - ((cone.n - 1) / u + sum(uden)) / u

    cone.hess_aux_updated = true
    return
end

function update_hess(cone::EpiNormInf{T}) where {T <: Real}
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

function update_inv_hess_aux(cone::EpiNormInf{T}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)
    @assert !cone.inv_hess_aux_updated
    wden = cone.wden
    u = cone.point[1]

    usqr = abs2(u)
    schur = (1 - cone.n) / usqr
    @inbounds for (j, wj) in enumerate(cone.w)
        u2pwj2 = T(0.5) * (usqr + abs2(wj))
        iedge = u / u2pwj2 * wj
        if cone.is_complex
            (cone.Hiure[j], cone.Hiuim[j]) = reim(iedge)
        else
            cone.Hiure[j] = iedge
        end
        schur += inv(u2pwj2)
    end
    cone.schur = schur

    if cone.is_complex
        @. cone.idet = cone.Hrere * cone.Himim - abs2(cone.Hreim)
    end

    cone.inv_hess_aux_updated = true
    return
end

function update_inv_hess(cone::EpiNormInf{T}) where {T <: Real}
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    if !isdefined(cone, :inv_hess)
        cone.inv_hess = Symmetric(zeros(T, cone.dim, cone.dim), :U)
    end
    Hi = cone.inv_hess.data
    wden = cone.wden
    u = cone.point[1]

    Hi[1, 1] = 1
    @inbounds for j in 1:cone.n
        if cone.is_complex
            Hi[1, 2j] = cone.Hiure[j]
            Hi[1, 2j + 1] = cone.Hiuim[j]
        else
            Hi[1, j + 1] = cone.Hiure[j]
        end
    end

    rtschur = sqrt(cone.schur)
    Hi[1, :] ./= rtschur
    @inbounds for j in 2:cone.dim, i in 2:j
        Hi[i, j] = Hi[1, j] * Hi[1, i]
    end
    Hi[1, :] ./= rtschur

    if cone.is_complex
        @inbounds for j in 1:cone.n
            detj = cone.idet[j]
            vj = 2j
            wj = vj + 1
            Hi[vj, vj] += cone.Himim[j] / detj
            Hi[wj, wj] += cone.Hrere[j] / detj
            Hi[vj, wj] -= cone.Hreim[j] / detj
        end
    else
        @inbounds for (j, rerej) in enumerate(cone.Hrere)
            vj = j + 1
            Hi[vj, vj] += inv(rerej)
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
        cone.no_sqrts = !factor_upper_arrow_block2(cone.Huu, cone.Hure, cone.Huim, cone.Hrere, cone.Hreim, cone.Himim, cone.hess_sqrt.data.nzval)
    else
        cone.no_sqrts = !factor_upper_arrow(cone.Huu, cone.Hure, cone.Hrere, cone.hess_sqrt.data.nzval)
    end

    cone.no_sqrts && println("no sqrt") # TODO remove later

    cone.hess_sqrt_aux_updated = true
    return
end

function hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormInf{T, T}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)

    @views @inbounds begin
        copyto!(prod[1, :], arr[1, :])
        mul!(prod[1, :], arr[2:end, :]', cone.Hure, true, cone.Huu)
        mul!(prod[2:end, :], cone.Hure, arr[1, :]')
        @. prod[2:end, :] += cone.Hrere * arr[2:end, :]
    end

    return prod
end

function hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormInf{T, Complex{T}}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)

    @views @inbounds begin
        @. prod[1, :] = cone.Huu * arr[1, :]
        mul!(prod[1, :], arr[2:2:end, :]', cone.Hure, true, true)
        mul!(prod[1, :], arr[3:2:end, :]', cone.Huim, true, true)
        mul!(prod[2:2:end, :], cone.Hure, arr[1, :]')
        mul!(prod[3:2:end, :], cone.Huim, arr[1, :]')
        @. prod[2:2:end, :] += cone.Hrere * arr[2:2:end, :] + cone.Hreim * arr[3:2:end, :]
        @. prod[3:2:end, :] += cone.Himim * arr[3:2:end, :] + cone.Hreim * arr[2:2:end, :]
    end

    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormInf{T, T}) where {T <: Real}
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)

    @views @inbounds begin
        copyto!(prod[1, :], arr[1, :])
        mul!(prod[1, :], arr[2:end, :]', cone.Hiure, true, true)
        @. prod[2:end, :] = cone.Hiure * prod[1, :]'
        prod ./= cone.schur
        @. prod[2:end, :] += arr[2:end, :] / cone.Hrere
    end

    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormInf{T, Complex{T}}) where {T <: Real}
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)

    @views @inbounds begin
        copyto!(prod[1, :], arr[1, :])
        mul!(prod[1, :], arr[2:2:end, :]', cone.Hiure, true, true)
        mul!(prod[1, :], arr[3:2:end, :]', cone.Hiuim, true, true)
        @. prod[2:2:end, :] = cone.Hiure * prod[1, :]'
        @. prod[3:2:end, :] = cone.Hiuim * prod[1, :]'
        prod ./= cone.schur
    end
    @views @inbounds for j in 1:cone.n
        j2 = 2j
        @. prod[j2, :] += (cone.Himim[j] * arr[j2, :] - cone.Hreim[j] * arr[j2 + 1, :]) / cone.idet[j]
        @. prod[j2 + 1, :] += (cone.Hrere[j] * arr[j2 + 1, :] - cone.Hreim[j] * arr[j2, :]) / cone.idet[j]
    end

    return prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    cone.hess_sqrt_aux_updated || update_hess_sqrt_aux(cone)
    @assert !cone.no_sqrts
    copyto!(prod, arr)
    lmul!(cone.hess_sqrt', prod)
    return prod
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    cone.hess_sqrt_aux_updated || update_hess_sqrt_aux(cone)
    @assert !cone.no_sqrts
    ldiv!(prod, cone.hess_sqrt, arr)
    return prod
end

function correction(cone::EpiNormInf{T}, primal_dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    u = cone.point[1]
    udir = primal_dir[1]
    corr = cone.correction

    u3 = T(1.5) / u
    udu = udir / u
    corr1 = -udir * sum(z * (u3 - z) * z for z in cone.uden) * udir - udu * (cone.n - 1) / u * udu
    @inbounds for i in 1:cone.n
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
