#=
utilities for polynomial interpolation on canonical domains
=#

sampling_region(dom::SemiFreeDomain) = dom.restricted_halfregion
sampling_region(dom::Domain) = dom


function interp_sample(dom::Box{T}, npts::Int) where {T <: Real}
    dim = get_dimension(dom)
    pts = rand(T, npts, dim) .- T(0.5)
    shift = (dom.u + dom.l) .* T(0.5)
    for i in 1:npts
        pts[i, :] = pts[i, :] .* (dom.u - dom.l) + shift
    end
    return pts
end

function get_weights(dom::Box{T}, pts::AbstractMatrix{T}) where {T <: Real}
    g = [(pts[:, i] .- dom.l[i]) .* (dom.u[i] .- pts[:, i]) for i in 1:size(pts, 2)]
    @assert all(all(gi .>= 0) for gi in g)
    return g
end


function interp_sample(dom::Ball{T}, npts::Int) where {T <: Real}
    dim = get_dimension(dom)
    pts = randn(T, npts, dim)
    norms = sum(abs2, pts, dims = 2)
    pts .*= dom.r ./ sqrt.(norms) # scale
    norms ./= 2
    pts .*= sf_gamma_inc_Q.(norms, dim / 2) .^ inv(dim) # sf_gamma_inc_Q is the normalized incomplete gamma function
    for i in 1:dim
        pts[:, i] .+= dom.c[i] # shift
    end
    return pts
end

function get_weights(dom::Ball{T}, pts::AbstractMatrix{T}) where {T <: Real}
    g = [abs2(dom.r) - sum((pts[j, :] - dom.c) .^ 2) for j in 1:size(pts, 1)]
    @assert all(g .>= 0)
    return [g]
end


function interp_sample(dom::Ellipsoid{T}, npts::Int) where {T <: Real}
    dim = get_dimension(dom)
    pts = randn(T, npts, dim)
    norms = sum(abs2, pts, dims = 2)
    for i in 1:npts
        pts[i, :] ./= sqrt(norms[i]) # scale
    end
    norms ./= 2
    pts .*= sf_gamma_inc_Q.(norms, dim / 2) .^ inv(dim) # sf_gamma_inc_Q is the normalized incomplete gamma function

    F_rotate_scale = cholesky(dom.Q).U
    for i in 1:npts
        pts[i, :] = F_rotate_scale \ pts[i, :] # rotate/scale
    end
    for i in 1:dim
        pts[:, i] .+= dom.c[i] # shift
    end

    return pts
end

function get_weights(dom::Ellipsoid{T}, pts::AbstractMatrix{T}) where {T <: Real}
    g = [1 - (pts[j, :] - dom.c)' * dom.Q * (pts[j, :] - dom.c) for j in 1:size(pts, 1)]
    @assert all(g .>= 0)
    return [g]
end

interp_sample(dom::SemiFreeDomain, npts::Int) =
    hcat(interp_sample(dom.restricted_halfregion, npts), interp_sample(dom.restricted_halfregion, npts))

get_weights(dom::SemiFreeDomain, pts::Matrix{<:Real}) =
    get_weights(dom.restricted_halfregion, view(pts, :, 1:div(size(pts, 2), 2)))


interp_sample(dom::FreeDomain{T}, npts::Int) where {T <: Real} = interp_sample(Box{T}(-ones(T, dom.n), ones(T, dom.n)), npts)

get_weights(::FreeDomain{T}, ::AbstractMatrix{T}) where {T <: Real} = T[]


get_L(n::Int, d::Int) = binomial(n + d, n)
get_U(n::Int, d::Int) = binomial(n + 2d, n)

function interpolate(
    dom::Domain{T},
    d::Int;
    calc_V::Bool = false,
    calc_w::Bool = false,
    sample = nothing,
    sample_factor::Int = 0,
    ) where {T <: Real}
    @assert d >= 1
    n = get_dimension(dom)
    U = binomial(n + 2d, n)

    if isnothing(sample)
        sample = (!isa(dom, Box) || n >= 7 || prod_consec(n, d) > 35_000)
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
                if U > 35_000 && calc_w
                    error("dimensions are too large to compute quadrature weights")
                end
            end
        end
        return wsos_sample_params(dom, d, calc_V, calc_w, sample_factor)
    else
        return wsos_box_params(sampling_region(dom), n, d, calc_V, calc_w)
    end
end

# slow but high-quality hyperrectangle/box point selections
wsos_box_params(dom::Domain, n::Int, d::Int, calc_V::Bool, calc_w::Bool) = error("non-sampling based interpolation methods are only available for box domains")

# difference with sampling functions is that P0 is always formed using points in [-1, 1]
function wsos_box_params(
    dom::Box{T},
    n::Int,
    d::Int,
    calc_V::Bool,
    calc_w::Bool,
    ) where {T <: Real}
    # n could be larger than the dimension of dom if the original domain was a SemiFreeDomain
    (U, pts, P0, P0sub, V, w) = wsos_box_params(T, n, d, calc_V, calc_w)

    # TODO refactor/cleanup below
    # scale and shift points, get WSOS matrices
    pscale = [T(0.5) * (dom.u[mod(j - 1, get_dimension(dom)) + 1] - dom.l[mod(j - 1, get_dimension(dom)) + 1]) for j in 1:n]
    pshift = [T(0.5) * (dom.u[mod(j - 1, get_dimension(dom)) + 1] + dom.l[mod(j - 1, get_dimension(dom)) + 1]) for j in 1:n]
    Wtsfun = (j -> sqrt.(1 .- abs2.(pts[:, j])) * pscale[j])
    PWts = [Wtsfun(j) .* P0sub for j in 1:get_dimension(dom)]
    trpts = pts .* pscale' .+ pshift'

    return (U = U, pts = trpts, Ps = [P0, PWts...], V = V, w = w)
end

function wsos_box_params(
    dom::FreeDomain{T},
    n::Int,
    d::Int,
    calc_V::Bool,
    calc_w::Bool,
    ) where {T <: Real}
    # n could be larger than the dimension of dom if the original domain was a SemiFreeDomain
    (U, pts, P0, P0sub, V, w) = wsos_box_params(T, n, d, calc_V, calc_w)
    return (U = U, pts = pts, Ps = [P0], V = V, w = w)
end

function wsos_box_params(T::Type{<:Real}, n::Int, d::Int, calc_V::Bool, calc_w::Bool)
    if n == 1
        return cheb2_data(T, d, calc_V, calc_w)
    elseif n == 2
        return padua_data(T, d, calc_V, calc_w) # or approxfekete_data(n, d)
    elseif n > 2
        return approxfekete_data(T, n, d, calc_V, calc_w)
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
    u = Matrix{T}(undef, length(pts_i), d + 1)
    @. u[:, 1] = 1
    @. u[:, 2] = pts_i
    for t in 3:(d + 1)
        @. @views u[:, t] = 2 * pts_i * u[:, t - 1] - u[:, t - 2]
    end

    if !calc_gradient && !calc_hessian
        return u
    end
    @assert calc_gradient

    # calculate gradient
    ug = similar(u)
    @. ug[:, 1] = 0
    @. ug[:, 2] = 1
    for t in 3:(d + 1)
        @. @views ug[:, t] = 2 * (u[:, t - 1] + pts_i * ug[:, t - 1]) - ug[:, t - 2]
    end

    if !calc_hessian
        return (u, ug)
    end
    @assert d > 1

    # calculate hessian
    uh = similar(u)
    @. uh[:, 1:2] = 0
    for t in 3:(d + 1)
        @. @views uh[:, t] = 2 * (2 * ug[:, t - 1] + pts_i * uh[:, t - 1]) - uh[:, t - 2]
    end

    return (u, ug, uh)
end

function cheb2_data(
    T::Type{<:Real},
    d::Int,
    calc_V::Bool,
    calc_w::Bool,
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
    if calc_w
        wa = T[2 / T(1 - j^2) for j in 0:2:(U - 1)]
        append!(wa, wa[div(U, 2):-1:2])
        w = real.(FFTW.ifft(wa))
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
    calc_w::Bool,
    )
    @assert d > 0
    U = get_U(2, d)
    L = get_L(2, d)

    # Padua points for degree 2d
    cheba = cheb2_pts(T, 2d + 1)
    chebb = cheb2_pts(T, 2d + 2)
    pts = Matrix{T}(undef, U, 2)
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
    if calc_w
        te1 = [cospi(T(i * j) / T(2d)) for i in 0:2:2d, j in 0:2:2d]
        to1 = [cospi(T(i * j) / T(2d)) for i in 0:2:2d, j in 1:2:2d]
        te2 = [cospi(T(i * j) / T(2d + 1)) for i in 0:2:2d, j in 0:2:(2d + 1)]
        to2 = [cospi(T(i * j) / T(2d + 1)) for i in 0:2:2d, j in 1:2:(2d + 1)]
        te1[2:(d + 1), :] .*= sqrt(T(2))
        to1[2:(d + 1), :] .*= sqrt(T(2))
        te2[2:(d + 1), :] .*= sqrt(T(2))
        to2[2:(d + 1), :] .*= sqrt(T(2))
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
        W = Matrix{T}(undef, d + 1, 2d + 1)
        W[:, 1:2:(2d + 1)] .= to2' * Mmom * te1
        W[:, 2:2:(2d + 1)] .= te2' * Mmom * to1
        W[:, [1, (2d + 1)]] ./= 2
        W[1, 2:2:(2d + 1)] ./= 2
        W[d + 1, 1:2:(2d + 1)] ./= 2
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
    calc_V::Bool,
    calc_w::Bool,
    )
    @assert d > 0
    @assert n > 1

    # points in the initial interpolation grid
    npts = prod_consec(n, d)
    candidate_pts = Matrix{T}(undef, npts, n)

    for j in 1:n
        ig = prod_consec(n, d, j)
        cs = cheb2_pts(T, 2d + j)
        i = 1
        l = 1
        while true
            candidate_pts[i:(i + ig - 1), j] .= cs[l]
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

    dom = Box{T}(-ones(T, n), ones(T, n))
    (pts, P0, P0sub, V, w) = make_wsos_arrays(dom, candidate_pts, d, calc_V, calc_w)

    return (size(pts, 1), pts, P0, P0sub, V, w)
end

# TODO could merge this function and choose_interp_pts
function make_wsos_arrays(
    dom::Domain{T},
    candidate_pts::Matrix{T},
    d::Int,
    calc_V::Bool,
    calc_w::Bool,
    ) where {T <: Real}
    n = size(candidate_pts, 2)
    (V, keep_pts, w) = choose_interp_pts(candidate_pts, d, calc_V, calc_w)
    pts = candidate_pts[keep_pts, :]
    P0 = V[:, 1:get_L(n, d)] # subset of polynomial evaluations up to total degree d
    Lsub = get_L(n, div(2d - get_degree(dom), 2))
    P0sub = view(P0, :, 1:Lsub)
    return (pts, P0, P0sub, V, w)
end

# sampling-based point selection for general domains
function wsos_sample_params(
    dom::Domain,
    d::Int,
    calc_V::Bool,
    calc_w::Bool,
    sample_factor::Int,
    )
    U = get_U(get_dimension(dom), d)
    candidate_pts = interp_sample(dom, U * sample_factor)
    (pts, P0, P0sub, V, w) = make_wsos_arrays(dom, candidate_pts, d, calc_V, calc_w)
    g = get_weights(dom, pts)
    PWts = [sqrt.(gi) .* P0sub for gi in g]
    return (U = U, pts = pts, Ps = [P0, PWts...], V = V, w = w)
end

n_deg_exponents(n::Int, deg::Int) = [xp for t in 0:deg for xp in Combinatorics.multiexponents(n, t)]

# indices of points to keep and quadrature weights at those points
function choose_interp_pts(
    candidate_pts::Matrix{T},
    d::Int,
    calc_V::Bool,
    calc_w::Bool,
    ) where {T <: Real}
    n = size(candidate_pts, 2)
    U = get_U(n, d)

    V = make_chebyshev_vandermonde(candidate_pts, 2d)

    if !calc_w && size(candidate_pts, 1) == U && U > 35_000
        # large matrix and don't need to perform QR procedure, so don't
        # TODO this is hacky; later the interpolate function and functions it calls should take options (and have better defaults) for whether to perform the QR or not
        return (V, 1:U, T[])
    end

    F = qr!(Array(V'), Val(true))
    keep_pts = F.p[1:U]
    V = V[keep_pts, :]

    if calc_w
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

# construct vandermonde with rows corresponding to points and columns to Chebyshev polys
function make_chebyshev_vandermonde(pts::Matrix{T}, deg::Int) where {T <: Real}
    n = size(pts, 2)
    expos = n_deg_exponents(n, deg)
    univ_chebs = [calc_univariate_chebyshev(view(pts, :, i), deg) for i in 1:n]
    return make_product_vandermonde(univ_chebs, expos)
end

function make_product_vandermonde(u::Vector{Matrix{T}}, expos::Vector) where {T <: Real}
    npts = size(u[1], 1)
    n = length(u)
    V = Matrix{T}(undef, npts, length(expos))

    for (col, xp) in enumerate(expos)
        @inbounds @. @views V[:, col] = u[1][:, xp[1] + 1]
        @inbounds for j in 2:n
            @. @views V[:, col] *= u[j][:, xp[j] + 1]
        end
    end

    return V
end


# these functions are not numerically stable for high degree (since they essentially calculate 2^degree); consider removing entirely
# TODO this is redundant if already do a QR of the U*U Vandermonde - just use that QR
function recover_lagrange_polys(pts::Matrix{T}, deg::Int) where {T <: Real}
    @warn("recover_lagrange_polys is not numerically stable for large degree")
    (U, n) = size(pts)
    DP.@polyvar x[1:n]
    basis = get_chebyshev_polys(x, deg)
    @assert length(basis) == U
    vandermonde_inv = inv([basis[j](x => view(pts, i, :)) for i in 1:U, j in 1:U])
    lagrange_polys = [DP.polynomial(view(vandermonde_inv, :, i), basis) for i in 1:U]
    return lagrange_polys
end

function calc_chebyshev_univariate(monovec::Vector{DynamicPolynomials.PolyVar{true}}, deg::Int)
    @warn("calc_u for polyvar input is not numerically stable for large degree")
    n = length(monovec)
    u = Vector{Vector}(undef, n)
    for j in 1:n
        uj = u[j] = Vector{DP.Polynomial{true, Int}}(undef, deg + 1)
        uj[1] = DP.Monomial(1)
        uj[2] = monovec[j]
        for t in 3:(deg + 1)
            uj[t] = 2 * uj[2] * uj[t - 1] - uj[t - 2]
        end
    end
    return u
end

# returns the multivariate Chebyshev polynomials in x up to degree deg
function get_chebyshev_polys(x::Vector{DynamicPolynomials.PolyVar{true}}, deg::Int)
    @warn("get_chebyshev_polys for polyvar input is not numerically stable for large degree")
    n = length(x)
    u = calc_chebyshev_univariate(x, deg)
    V = Vector{DP.Polynomial{true, Int}}(undef, get_L(n, deg))
    V[1] = DP.Monomial(1)
    col = 1
    for t in 1:deg, xp in Combinatorics.multiexponents(n, t)
        col += 1
        V[col] = u[1][xp[1] + 1]
        for j in 2:n
            V[col] *= u[j][xp[j] + 1]
        end
    end
    return V
end
