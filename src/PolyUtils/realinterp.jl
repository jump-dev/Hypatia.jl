#=
utilities for real polynomial interpolation
=#

"""
$(SIGNATURES)

Compute interpolation data for a real weighted sum-of-squares conic constraint
on a domain.
"""
function interpolate(
    dom::Domain{T},
    d::Int;
    calc_V::Bool = false,
    get_quadr::Bool = false,
    sample = nothing,
    sample_factor::Int = 0,
    ) where {T <: Real}
    @assert d >= 1
    n = dimension(dom)
    U = binomial(n + 2d, n)

    if isnothing(sample)
        sample = (!isa(dom, BoxDomain) || n >= 7 || prod_consec(n, d) > 35_000)
    end

    if sample
        if sample_factor <= 0
            if U <= 12_000
                sample_factor = 10
            elseif U <= 15_000
                sample_factor = 5
            elseif U <= 22_000
                sample_factor = 2
            else
                sample_factor = 1
                if U > 35_000 && get_quadr
                    error("dimensions are too large to compute quadrature weights")
                end
            end
        end
        return interp_sample(dom, d, get_quadr, sample_factor)
    else
        return interp_box(dom, n, d, calc_V, get_quadr)
    end
end

get_L(n::Int, d::Int) = binomial(n + d, n)
get_U(n::Int, d::Int) = binomial(n + 2d, n)

# sampling-based point selection for general domains
function interp_sample(
    dom::Domain{T},
    d::Int,
    get_quadr::Bool,
    sample_factor::Int,
    ) where {T <: Real}
    U = get_U(dimension(dom), d)

    cand_pts = sample(dom, U * sample_factor)
    (pts, P0, P0sub, V, w) = make_wsos_arrays(dom, cand_pts, d, get_quadr)

    g = weights(dom, pts)
    PWts = Matrix{T}[sqrt.(gi) .* P0sub for gi in g]
    Ps = Matrix{T}[P0, PWts...]

    return (U = U, pts = pts, Ps = Ps, V = V, w = w)
end

# slow but high-quality hyperrectangle/box point selections
# unlike sampling functions, P0 is always formed using points in [-1, 1]
function interp_box(
    dom::FreeDomain{T},
    n::Int,
    d::Int,
    calc_V::Bool,
    get_quadr::Bool,
    ) where {T <: Real}
    (U, pts, P0, P0sub, V, w) = interp_box(T, n, d, calc_V, get_quadr)
    return (U = U, pts = pts, Ps = Matrix{T}[P0,], V = V, w = w)
end

function interp_box(
    dom::BoxDomain{T},
    n::Int,
    d::Int,
    calc_V::Bool,
    get_quadr::Bool,
    ) where {T <: Real}
    (U, pts, P0, P0sub, V, w) = interp_box(T, n, d, calc_V, get_quadr)

    # TODO refactor/cleanup below
    # scale and shift points, get WSOS matrices
    pscale = [T(0.5) * (dom.u[mod(j - 1, dimension(dom)) + 1] -
        dom.l[mod(j - 1, dimension(dom)) + 1]) for j in 1:n]
    pshift = [T(0.5) * (dom.u[mod(j - 1, dimension(dom)) + 1] +
        dom.l[mod(j - 1, dimension(dom)) + 1]) for j in 1:n]
    @views Wtsfun = (j -> sqrt.(1 .- abs2.(pts[:, j])) * pscale[j])
    PWts = Matrix{T}[Wtsfun(j) .* P0sub for j in 1:dimension(dom)]
    trpts = pts .* pscale' .+ pshift'

    return (U = U, pts = trpts, Ps = Matrix{T}[P0, PWts...], V = V, w = w)
end

function interp_box(
    T::Type{<:Real},
    n::Int,
    d::Int,
    calc_V::Bool,
    get_quadr::Bool,
    )
    if n == 1
        return cheb2_data(T, d, calc_V, get_quadr)
    elseif n == 2
        return padua_data(T, d, calc_V, get_quadr) # or approxfekete_data(n, d)
    elseif n > 2
        return approxfekete_data(T, n, d, get_quadr)
    end
end

# return k Chebyshev points of the second kind
cheb2_pts(T::Type{<:Real}, k::Int) = [-cospi(T(j) / T(k - 1)) for j in 0:(k - 1)]

function calc_univariate_chebyshev(
    pts_i::AbstractVector{T},
    d::Int;
    calc_gradient::Bool = false,
    calc_hessian::Bool = false,
    ) where {T <: Real}
    @assert d > 0
    u = zeros(T, length(pts_i), d + 1)
    @. @views u[:, 1] = 1
    @. @views u[:, 2] = pts_i
    for t in 3:(d + 1)
        @. @views u[:, t] = 2 * pts_i * u[:, t - 1] - u[:, t - 2]
    end

    if !calc_gradient && !calc_hessian
        return u
    end
    @assert calc_gradient

    # calculate gradient
    ug = zero(u)
    @. @views ug[:, 1] = 0
    @. @views ug[:, 2] = 1
    for t in 3:(d + 1)
        @. @views ug[:, t] = 2 * (u[:, t - 1] + pts_i *
            ug[:, t - 1]) - ug[:, t - 2]
    end

    if !calc_hessian
        return (u, ug)
    end
    @assert d > 1

    # calculate hessian
    uh = zero(u)
    @. @views uh[:, 1:2] = 0
    for t in 3:(d + 1)
        @. @views uh[:, t] = 2 * (2 * ug[:, t - 1] + pts_i *
            uh[:, t - 1]) - uh[:, t - 2]
    end

    return (u, ug, uh)
end

function cheb2_data(
    T::Type{<:Real},
    d::Int,
    calc_V::Bool,
    get_quadr::Bool,
    )
    @assert d > 0
    U = get_U(1, d)
    L = get_L(1, d)

    # Chebyshev points for degree 2d
    pts = reshape(cheb2_pts(T, U), :, 1)

    # evaluations
    if calc_V
        V = make_chebyshev_vandermonde(pts, 2d)
        P0 = V[:, 1:L]
    else
        V = zeros(T, 0, 0)
        P0 = make_chebyshev_vandermonde(pts, d)
    end
    P0sub = view(P0, :, 1:get_L(1, d - 1))

    # weights for Clenshaw-Curtis quadrature at pts
    if get_quadr
        wa = T[2 / T(1 - j^2) for j in 0:2:(U - 1)]
        @views append!(wa, wa[div(U, 2):-1:2])
        tempconst = pi / T(length(wa)) * 2 * im
        # inverse FFT
        w = [abs(sum(wa[j] * exp(tempconst * (i - 1) * j) for
            j in eachindex(wa)) / length(wa)) for i in eachindex(wa)]
        w[1] /= 2
        push!(w, w[1])
    else
        w = T[]
    end

    return (U, pts, P0, P0sub, V, w)
end

function padua_data(
    T::Type{<:Real},
    d::Int,
    calc_V::Bool,
    get_quadr::Bool,
    )
    @assert d > 0
    U = get_U(2, d)
    L = get_L(2, d)

    # Padua points for degree 2d
    cheba = cheb2_pts(T, 2d + 1)
    chebb = cheb2_pts(T, 2d + 2)
    pts = zeros(T, U, 2)
    j = 1
    for a in 0:2d, b in 0:(2d + 1)
        if iseven(a + b)
            pts[j, 1] = -cheba[a + 1]
            pts[(U + 1 - j), 2] = -chebb[2d + 2 - b]
            j += 1
        end
    end

    # evaluations
    if calc_V
        V = make_chebyshev_vandermonde(pts, 2d)
        P0 = V[:, 1:L]
    else
        V = zeros(T, 0, 0)
        P0 = make_chebyshev_vandermonde(pts, d)
    end
    P0sub = view(P0, :, 1:get_L(2, d - 1))

    # cubature weights at Padua points
    # even-degree Chebyshev polynomials on the subgrids
    if get_quadr
        te1 = [cospi(T(i * j) / T(2d)) for i in 0:2:2d, j in 0:2:2d]
        to1 = [cospi(T(i * j) / T(2d)) for i in 0:2:2d, j in 1:2:2d]
        te2 = [cospi(T(i * j) / T(2d + 1)) for i in 0:2:2d, j in 0:2:(2d + 1)]
        to2 = [cospi(T(i * j) / T(2d + 1)) for i in 0:2:2d, j in 1:2:(2d + 1)]
        @views te1[2:(d + 1), :] .*= sqrt(T(2))
        @views to1[2:(d + 1), :] .*= sqrt(T(2))
        @views te2[2:(d + 1), :] .*= sqrt(T(2))
        @views to2[2:(d + 1), :] .*= sqrt(T(2))
        # even, even moments matrix
        mom = T(2) * sqrt(T(2)) ./ [T(1 - i^2) for i in 0:2:2d]
        mom[1] = 2
        Mmom = zeros(T, d + 1, d + 1)
        f = inv(T(d * (2d + 1)))
        for j in 1:(d + 1), i in 1:(d + 2 - j)
            Mmom[i, j] = mom[i] * mom[j] * f
        end
        Mmom[1, d + 1] /= 2
        # cubature weights as matrices on the subgrids
        W = zeros(T, d + 1, 2d + 1)
        @views W[:, 1:2:(2d + 1)] .= to2' * Mmom * te1
        @views W[:, 2:2:(2d + 1)] .= te2' * Mmom * to1
        @views W[:, [1, (2d + 1)]] ./= 2
        @views W[1, 2:2:(2d + 1)] ./= 2
        @views W[d + 1, 1:2:(2d + 1)] ./= 2
        w = vec(W)
    else
        w = T[]
    end

    return (U, pts, P0, P0sub, V, w)
end

prod_consec(n::Int, d::Int, j::Int = 0) = prod(big(2d + 1 + j):big(2d + n))

function approxfekete_data(
    T::Type{<:Real},
    n::Int,
    d::Int,
    get_quadr::Bool,
    )
    @assert d > 0
    @assert n > 1

    # points in the initial interpolation grid
    npts = prod_consec(n, d)
    cand_pts = zeros(T, npts, n)

    for j in 1:n
        ig = prod_consec(n, d, j)
        cs = cheb2_pts(T, 2d + j)
        i = 1
        l = 1
        while true
            @views cand_pts[i:(i + ig - 1), j] .= cs[l]
            i += ig
            l += 1
            if l >= 2d + 1 + j
                if i >= npts
                    break
                end
                l = 1
            end
        end
    end

    dom = BoxDomain{T}(-ones(T, n), ones(T, n))
    (pts, P0, P0sub, V, w) = make_wsos_arrays(dom, cand_pts, d, get_quadr)

    return (size(pts, 1), pts, P0, P0sub, V, w)
end

# TODO could merge this function and choose_interp_pts
function make_wsos_arrays(
    dom::Domain{T},
    cand_pts::Matrix{T},
    d::Int,
    get_quadr::Bool,
    ) where {T <: Real}
    n = size(cand_pts, 2)
    (V, keep_pts, w) = choose_interp_pts(cand_pts, d, get_quadr)
    pts = cand_pts[keep_pts, :]
    P0 = V[:, 1:get_L(n, d)] # subset of poly evaluations up to total degree d
    Lsub = get_L(n, div(2d - degree(dom), 2))
    P0sub = view(P0, :, 1:Lsub)
    return (pts, P0, P0sub, V, w)
end

n_deg_exponents(n::Int, deg::Int) = [xp for t in 0:deg for
    xp in Combinatorics.multiexponents(n, t)]

# indices of points to keep and quadrature weights at those points
function choose_interp_pts(
    cand_pts::Matrix{T},
    d::Int,
    get_quadr::Bool,
    ) where {T <: Real}
    n = size(cand_pts, 2)
    U = get_U(n, d)

    V = make_chebyshev_vandermonde(cand_pts, 2d)

    if !get_quadr && size(cand_pts, 1) == U && U > 35_000
        # large matrix and don't need to perform QR procedure, so don't
        # TODO this is hacky; later the interpolate function and functions it
        # calls should take options (and have better defaults) for whether to
        # perform the QR or not
        return (V, 1:U, T[])
    end

    F = qr!(Array(V'), ColumnNorm())
    keep_pts = F.p[1:U]
    V = V[keep_pts, :]

    if get_quadr
        n > 32 && @warn("quadrature weights not numerically stable for large n")
        m = zeros(T, U)
        for (col, xp) in enumerate(n_deg_exponents(n, 2d))
            if all(iseven, xp)
                @inbounds m[col] = prod(2 / (1 - xp[j]^2) for j in 1:n)
            end
        end
        Qtm = F.Q' * m
        w = UpperTriangular(F.R[:, 1:U]) \ Qtm
        return (V, keep_pts, w)
    end

    return (V, keep_pts, T[])
end

# make Vandermonde with rows corresponding to points and columns to Chebyshev polys
function make_chebyshev_vandermonde(pts::Matrix{T}, deg::Int) where {T <: Real}
    n = size(pts, 2)
    expos = n_deg_exponents(n, deg)
    univ_chebs = [calc_univariate_chebyshev(view(pts, :, i), deg) for i in 1:n]
    return make_product_vandermonde(univ_chebs, expos)
end

function make_product_vandermonde(
    u::Vector{Matrix{T}},
    expos::Vector,
    ) where {T <: Real}
    npts = size(u[1], 1)
    n = length(u)
    V = zeros(T, npts, length(expos))

    for (col, xp) in enumerate(expos)
        @inbounds @. @views V[:, col] = u[1][:, xp[1] + 1]
        @inbounds for j in 2:n
            @. @views V[:, col] *= u[j][:, xp[j] + 1]
        end
    end

    return V
end
