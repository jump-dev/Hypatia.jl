"""
$(TYPEDEF)

Epigraph of real or complex matrix spectral norm (i.e. maximum singular value)
for a matrix (stacked column-wise) of `nrows` rows and `ncols` columns with
`nrows ≤ ncols`.

    $(FUNCTIONNAME){T, R}(nrows::Int, ncols::Int, use_dual::Bool = false)
"""
mutable struct EpiNormSpectral{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d1::Int
    d2::Int
    is_complex::Bool

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

    W_svd
    s::Vector{T}
    U::Matrix{R}
    Vt::Matrix{R}
    z::Vector{T}
    trZi::T
    trZi2::T
    trZi3::T
    cu::T

    temp12::Matrix{R}

    function EpiNormSpectral{T, R}(
        d1::Int,
        d2::Int;
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert 1 <= d1 <= d2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.is_complex = (R <: Complex)
        cone.dim = 1 + vec_length(R, d1 * d2)
        cone.d1 = d1
        cone.d2 = d2
        return cone
    end
end

reset_data(cone::EpiNormSpectral) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

# TODO only allocate the fields we use
function setup_extra_data!(
    cone::EpiNormSpectral{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    dim = cone.dim
    (d1, d2) = (cone.d1, cone.d2)
    cone.z = zeros(T, d1)

    cone.temp12 = zeros(R, d1, d2)

    # cone.Z = zeros(R, d1, d1)
    # cone.Zi = zeros(R, d1, d1)
    # cone.SZi = zeros(R, d1, d2)
    # cone.HuW = zeros(R, d1, d2)
    # cone.S2Zi = zeros(R, d2, d2)
    # cone.ZiSZi = zeros(R, d1, d2)
    # cone.temp12b = zeros(R, d1, d2)
    # cone.temp12c = zeros(R, d1, d2)
    # cone.temp12d = zeros(R, d1, d2)
    # cone.tempd1d1 = zeros(R, d1, d1)
    # cone.tempd2d2 = zeros(R, d2, d2)
    # cone.tempd2d2b = zeros(R, d2, d2)
    return cone
end

get_nu(cone::EpiNormSpectral) = cone.d1 + 1

function set_initial_point!(
    arr::AbstractVector{T},
    cone::EpiNormSpectral{T},
    ) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

# TODO speed up using cholesky?
# TODO also use norm bound
function update_feas(cone::EpiNormSpectral{T}) where {T <: Real}
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views vec_copyto!(cone.temp12, cone.point[2:end])
        cone.W_svd = svd(cone.temp12, full = false) # TODO in place
        cone.is_feas = (u - maximum(cone.W_svd.S) > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

# TODO speed up using cholesky?
# TODO also use norm bound
function is_dual_feas(cone::EpiNormSpectral{T}) where {T <: Real}
    u = cone.dual_point[1]
    if u > eps(T)
        W = @views vec_copyto!(cone.temp12, cone.dual_point[2:end])
        return (u - sum(svdvals!(W)) > eps(T))
    end
    return false
end

function update_grad(cone::EpiNormSpectral)
    @assert cone.is_feas
    u = cone.point[1]
    s = cone.s = cone.W_svd.S
    U = cone.U = cone.W_svd.U
    Vt = cone.Vt = cone.W_svd.Vt
    z = cone.z
    g = cone.grad

    @. z = abs2(u) - abs2(s)
    cone.trZi = sum(inv, z)
    cone.cu = (cone.d1 - 1) / u
    # cone.szi =

    g[1] = -2 * u * cone.trZi + cone.cu
    szi2 = 2 * s ./ z
    gW = U * Diagonal(szi2) * Vt
    @views vec_copyto!(g[2:end], gW)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiNormSpectral)
    # cone.hess_aux_updated || update_hess_aux(cone)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    H = cone.hess.data
    u = cone.point[1]
    s = cone.s
    U = cone.U
    Vt = cone.Vt
    z = cone.z

    trZi2 = sum(zi -> abs2(inv(zi)), z)
    Huu = 4 * abs2(u) * trZi2 - 2 * cone.trZi - cone.cu / u
    huw = -4 * u * s ./ abs2.(z)
    Huw = U * Diagonal(huw) * Vt

    # u, w
    H[1, 1] = Huu
    @views vec_copyto!(H[1, 2:end], Huw)

    # w, w
    U_irtz = U * Diagonal(inv.(sqrt.(z)))
    sirtz_Vt = Diagonal(s ./ sqrt.(z)) * Vt
    Zi = U_irtz * U_irtz'
    SZi = U_irtz * sirtz_Vt
    S2Zi = sirtz_Vt' * sirtz_Vt + I # TODO d2^2, don't allocate unless allocating hess

    idx_incr = (cone.is_complex ? 2 : 1)
    r_idx = 2
    for i in 1:d2, j in 1:d1
        c_idx = r_idx
        for k in i:d2
            S2Ziik = 2 * S2Zi[i, k]
            SZijk = 2 * SZi[j, k]
            lstart = (i == k ? j : 1)
            for l in lstart:d1
                term1 = Zi[l, j] * S2Ziik
                term2 = SZi[l, i] * SZijk
                spectral_kron_element!(H, r_idx, c_idx, term1, term2)
                c_idx += idx_incr
            end
        end
        r_idx += idx_incr
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectral,
    )
    @assert cone.grad_updated
    u = cone.point[1]
    s = cone.s
    U = cone.U
    Vt = cone.Vt
    z = cone.z
    temp12 = cone.temp12

    # TODO refac hess
    zi = inv.(z)
    trZi2 = sum(abs2, zi)
    uzi = u ./ z
    Huu = 4 * sum(abs2, uzi) - 2 * cone.trZi - cone.cu / u
    szi = s ./ z
    huw = -4 * uzi .* szi
    Uzi = U * (2 * Diagonal(zi)) # factor 2
    sziVt = Diagonal(szi) * Vt

    @inbounds for j in 1:size(prod, 2)
        arr_1j = arr[1, j]
        @views vec_copyto!(temp12, arr[2:end, j])
        # sim
        simU = Uzi' * temp12
        sim = simU * sziVt'
        # u
        @inbounds prod[1, j] = Huu * arr_1j - 2 * u *
            sum(real(sim[i, i]) for i in eachindex(s))
        # W
        HW1 = Diagonal(huw * arr_1j) + (sim + sim') * Diagonal(s)
        HW = U * (simU + HW1 * Vt)
        @views vec_copyto!(prod[2:end, j], HW)
    end

    return prod
end

function update_inv_hess(cone::EpiNormSpectral{T}) where {T <: Real}
    # cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    @assert cone.grad_updated
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    Hi = cone.inv_hess.data
    u = cone.point[1]
    s = cone.s
    U = cone.U
    Vt = cone.Vt
    z = cone.z
    temp12 = cone.temp12

    u2 = abs2(u)
    t = u2 .+ abs2.(s)
    sigma = -cone.cu / u + 2 * sum(inv, t)
    hiuw = 2 * u / sigma * s ./ t
    Hiuw = U * Diagonal(hiuw) * Vt
    hiuw2 = 8 * u2 / sigma ./ (t .* z)

    Den = zeros(T, d1, d1)
    for j in 1:d1
        s_j = s[j]
        for i in 1:d1
            sij = s[i] * s_j
            Den[i, j] = z[i] * s_j / ((u2 - sij) * (u2 + sij))
        end
        Den[j, j] = s_j / t[j]
    end

    ziUt = (T(0.5) * Diagonal(z)) * U'
    Vs = Diagonal(s) * Vt

    # u, w
    Hi[1, 1] = inv(sigma)
    @views vec_copyto!(Hi[1, 2:end], Hiuw)

    # w, w
    c_idx = 2
    reim1s = (cone.is_complex ? [1, im] : [1,])
    for i in 1:d2, j in 1:d1, reim1 in reim1s
        ziUtj = reim1 * ziUt[:, j]
        sim = ziUtj * Vs[:, i]'
        dconst = dot(hiuw2, real.(diag(sim)))
        HiW1 = Den .* (dconst * I - (sim + sim')) # TODO @. ?
        tempij = HiW1 * Vt
        tempij[:, i] += ziUtj
        HiW = U * tempij
        @views vec_copyto!(Hi[2:end, c_idx], HiW)
        c_idx += 1
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormSpectral{T},
    ) where T
    @assert cone.grad_updated
    u = cone.point[1]
    s = cone.s
    U = cone.U
    Vt = cone.Vt
    z = cone.z
    temp12 = cone.temp12

    u2 = abs2(u)
    t = u2 .+ abs2.(s)
    sigma = -cone.cu / u + 2 * sum(inv, t)
    hiuw2 = 4 * u / sigma ./ (t .* z)

    Uzi = U * (T(0.5) * Diagonal(z)) # factor 1/2
    sVt = Diagonal(s) * Vt

    d1 = cone.d1
    Den = zeros(T, d1, d1)
    for j in 1:d1
        s_j = s[j]
        for i in 1:d1
            sij = s[i] * s_j
            Den[i, j] = z[i] * s_j / ((u2 - sij) * (u2 + sij))
        end
        Den[j, j] = s_j / t[j]
    end

    @inbounds for j in 1:size(prod, 2)
        arr_1j = arr[1, j]
        @views vec_copyto!(temp12, arr[2:end, j])
        # sim
        simU = Uzi' * temp12
        sim = simU * sVt'
        c1 = arr_1j / sigma +
            sum(hiuw2[i] * real(sim[i, i]) for i in eachindex(s))
        # u
        prod[1, j] = c1
        # W
        dconst = 2 * u * c1
        HiW1 = Den .* (dconst * I - (sim + sim'))
        HiW = U * (simU + HiW1 * Vt)
        @views vec_copyto!(prod[2:end, j], HiW)
    end

    return prod
end

function dder3(cone::EpiNormSpectral, dir::AbstractVector)
    @assert cone.grad_updated
    u = cone.point[1]
    s = cone.s
    U = cone.U
    Vt = cone.Vt
    z = cone.z
    dder3 = cone.dder3

    u_dir = dir[1]
    @views W_dir = vec_copyto!(cone.temp12, dir[2:end])

    u2 = abs2(u)
    zi = inv.(z)
    c1 = abs2.(2 * u ./ z) - zi
    udzi = u_dir ./ z
    szi = s ./ z
    Ds = Diagonal(s)
    Dzi = Diagonal(zi)
    Dszi = Diagonal(szi)

    simU = U' * W_dir
    sims = (simU * Vt') * Ds

    M3 = sims + sims'
    M2 = Hermitian(Dzi * M3 * Dzi)

    M4a = Dzi * M2
    D1 = Diagonal(udzi .* c1)
    M6 = -2 * u * Hermitian(M4a + M4a') + D1
    tr2 = real(dot(sims, M6 + 3 * D1))

    simUa = Dzi * simU
    M5 = Hermitian(simUa * simUa', :U)
    tr1 = tr(M5)

    M5 += u_dir * M6
    M7 = M2 * M3 * Dszi + M5 * Ds

    M2 += Diagonal(-2 * u * udzi .* zi)
    M8 = M2 * simU

    Wcorr = -2 * U * (M8 + M7 * Vt)
    @views vec_copyto!(dder3[2:end], Wcorr)

    c1 .-= 2 * zi
    dder3[1] = 2 * u * (tr1 + u_dir * sum(udzi .* c1)) -
        cone.cu * abs2(u_dir / u) - tr2

    return dder3
end
