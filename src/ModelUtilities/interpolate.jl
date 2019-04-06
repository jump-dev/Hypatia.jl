#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
Copyright 2018, David Papp, Sercan Yildiz

utilities for interpolation on canonical domains

modified/inspired from files in https://github.com/dpapp-github/alfonso/
- https://github.com/dpapp-github/alfonso/blob/master/ChebInterval.m
- https://github.com/dpapp-github/alfonso/blob/master/PaduaSquare.m
- https://github.com/dpapp-github/alfonso/blob/master/FeketeCube.m
and Matlab files in the packages
- Chebfun http://www.chebfun.org/
- Padua2DM by M. Caliari, S. De Marchi, A. Sommariva, and M. Vianello http://profs.sci.univr.it/~caliari/software.htm
=#

sampling_region(dom::SemiFreeDomain) = dom.restricted_halfregion
sampling_region(dom::Domain) = dom


function interp_sample(dom::Box, npts::Int)
    dim = get_dimension(dom)
    pts = rand(npts, dim) .- 0.5
    shift = (dom.u + dom.l) .* 0.5
    for i in 1:npts
        pts[i, :] = pts[i, :] .* (dom.u - dom.l) + shift
    end
    return pts
end

function get_weights(dom::Box, pts::AbstractMatrix{Float64})
    g = [(pts[:, i] .- dom.l[i]) .* (dom.u[i] .- pts[:, i]) for i in 1:size(pts, 2)]
    @assert all(all(gi .>= 0.0) for gi in g)
    return g
end


function interp_sample(dom::Ball, npts::Int)
    dim = get_dimension(dom)
    pts = randn(npts, dim)
    norms = sum(abs2, pts, dims = 2)
    pts .*= dom.r ./ sqrt.(norms) # scale
    norms .*= 0.5
    pts .*= sf_gamma_inc_Q.(norms, dim * 0.5) .^ inv(dim) # sf_gamma_inc_Q is the normalized incomplete gamma function
    for i in 1:dim
        pts[:, i] .+= dom.c[i] # shift
    end
    return pts
end

function get_weights(dom::Ball, pts::AbstractMatrix{Float64})
    g = [dom.r^2 - sum((pts[j, :] - dom.c) .^ 2) for j in 1:size(pts, 1)]
    @assert all(g .>= 0.0)
    return [g]
end


function interp_sample(dom::Ellipsoid, npts::Int)
    dim = get_dimension(dom)
    pts = randn(npts, dim)
    norms = sum(abs2, pts, dims = 2)
    for i in 1:npts
        pts[i, :] ./= sqrt(norms[i]) # scale
    end
    norms .*= 0.5
    pts .*= sf_gamma_inc_Q.(norms, dim * 0.5) .^ inv(dim) # sf_gamma_inc_Q is the normalized incomplete gamma function

    F_rotate_scale = cholesky(dom.Q).U
    for i in 1:npts
        pts[i, :] = F_rotate_scale \ pts[i, :] # rotate/scale
    end

    for i in 1:dim
        pts[:, i] .+= dom.c[i] # shift
    end
    return pts
end

function get_weights(dom::Ellipsoid, pts::AbstractMatrix{Float64})
    g = [1.0 - (pts[j, :] - dom.c)' * dom.Q * (pts[j, :] - dom.c) for j in 1:size(pts, 1)]
    @assert all(g .>= 0.0)
    return [g]
end


interp_sample(dom::SemiFreeDomain, npts::Int) =
    hcat(interp_sample(dom.restricted_halfregion, npts), interp_sample(dom.restricted_halfregion, npts))

get_weights(dom::SemiFreeDomain, pts::Matrix{Float64}) =
    get_weights(dom.restricted_halfregion, view(pts, :, 1:div(size(pts, 2), 2)))


interp_sample(dom::FreeDomain, npts::Int) = interp_sample(Box(-ones(dom.n), ones(dom.n)), npts)

get_weights(::FreeDomain, ::AbstractMatrix{Float64}) = []


get_L_U(n::Int, d::Int) = (binomial(n + d, n), binomial(n + 2d, n))

function interpolate(
    dom::Domain,
    d::Int; # TODO make this 2d
    sample::Bool = false,
    calc_w::Bool = false,
    sample_factor::Int = 10,
    )
    if sample
        return wsos_sample_params(dom, d, calc_w = calc_w, sample_factor = sample_factor)
    else
        return wsos_box_params(sampling_region(dom), get_dimension(dom), d, calc_w = calc_w)
    end
end

# slow but high-quality hyperrectangle/box point selections
wsos_box_params(dom::Domain, n::Int, d::Int; calc_w::Bool) = error("accurate methods for interpolation points are only available for box domains, use sampling instead")

# difference with sampling functions is that P0 is always formed using points in [-1, 1]
function wsos_box_params(dom::Box, n::Int, d::Int; calc_w::Bool = false)
    # n could be larger than the dimension of dom if the original domain was a SemiFreeDomain
    (U, pts, P0, P0sub, w) = wsos_box_params(n, d, calc_w)

    # TODO refactor/cleanup below
    # scale and shift points, get WSOS matrices
    pscale = [0.5 * (dom.u[mod(j - 1, get_dimension(dom)) + 1] - dom.l[mod(j - 1,get_dimension(dom)) + 1]) for j in 1:n]
    pshift = [0.5 * (dom.u[mod(j - 1, get_dimension(dom)) + 1] + dom.l[mod(j - 1,get_dimension(dom)) + 1]) for j in 1:n]
    Wtsfun = (j -> sqrt.(1.0 .- abs2.(pts[:, j])) * pscale[j])
    PWts = [Wtsfun(j) .* P0sub for j in 1:get_dimension(dom)]
    trpts = pts .* pscale' .+ pshift'

    return (U = U, pts = trpts, P0 = P0, PWts = PWts, w = w)
end

function wsos_box_params(dom::FreeDomain, n::Int, d::Int; calc_w::Bool = false)
    # n could be larger than the dimension of dom if the original domain was a SemiFreeDomain
    (U, pts, P0, P0sub, w) = wsos_box_params(n, d, calc_w)
    return (U = U, pts = pts, P0 = P0, PWts = [], w = w)
end

function wsos_box_params(n::Int, d::Int, calc_w::Bool)
    if n == 1
        return cheb2_data(d, calc_w)
    elseif n == 2
        return padua_data(d, calc_w) # or approxfekete_data(n, d)
    elseif n > 2
        return approxfekete_data(n, d, calc_w)
    end
end

# return k Chebyshev points of the second kind
cheb2_pts(k::Int) = [-cospi(j / (k - 1)) for j in 0:(k - 1)]

function calc_u(n::Int, d::Int, pts::Matrix{Float64})
    @assert d > 0
    u = Vector{Matrix{Float64}}(undef, n)
    for j in 1:n
        uj = u[j] = Matrix{Float64}(undef, size(pts, 1), d + 1)
        uj[:, 1] .= 1.0
        @. @views uj[:, 2] = pts[:, j]
        for t in 3:(d + 1)
            @. @views uj[:, t] = 2.0 * uj[:, 2] * uj[:, t - 1] - uj[:, t - 2]
        end
    end
    return u
end

function cheb2_data(d::Int, calc_w::Bool)
    @assert d > 0
    U = 2d + 1

    # Chebyshev points for degree 2d
    pts = reshape(cheb2_pts(U), :, 1)

    # evaluations
    P0 = calc_u(1, d, pts)[1]
    P0sub = view(P0, :, 1:d)

    # weights for Clenshaw-Curtis quadrature at pts
    if calc_w
        wa = [2.0 / (1 - j^2) for j in 0:2:(U - 1)]
        append!(wa, wa[div(U, 2):-1:2])
        w = real.(FFTW.ifft(wa))
        w[1] *= 0.5
        push!(w, w[1])
    else
        w = Float64[]
    end

    return (U, pts, P0, P0sub, w)
end

function padua_data(d::Int, calc_w::Bool)
    @assert d > 0
    (L, U) = get_L_U(2, d)

    # Padua points for degree 2d
    cheba = cheb2_pts(2d + 1)
    chebb = cheb2_pts(2d + 2)
    pts = Matrix{Float64}(undef, U, 2)
    j = 1
    for a in 0:2d
        for b in 0:2d+1
            if iseven(a + b)
                pts[j, 1] = -cheba[a + 1]
                pts[(U + 1 - j), 2] = -chebb[2d + 2 - b]
                j += 1
            end
        end
    end

    # evaluations
    u = calc_u(2, d, pts)
    P0 = Matrix{Float64}(undef, U, L)
    P0[:, 1] .= 1
    col = 1
    for t in 1:d
        for xp in Combinatorics.multiexponents(2, t)
            col += 1
            P0[:, col] .= u[1][:, xp[1] + 1] .* u[2][:, xp[2] + 1]
        end
    end
    P0sub = view(P0, :, 1:binomial(1 + d, 2))

    # cubature weights at Padua points
    # even-degree Chebyshev polynomials on the subgrids
    if calc_w
        te1 = [cospi(i * j / (2d)) for i in 0:2:2d, j in 0:2:2d]
        to1 = [cospi(i * j / (2d)) for i in 0:2:2d, j in 1:2:2d]
        te2 = [cospi(i * j / (2d + 1)) for i in 0:2:2d, j in 0:2:(2d + 1)]
        to2 = [cospi(i * j / (2d + 1)) for i in 0:2:2d, j in 1:2:(2d + 1)]
        te1[2:(d + 1), :] .*= sqrt(2)
        to1[2:(d + 1), :] .*= sqrt(2)
        te2[2:(d + 1), :] .*= sqrt(2)
        to2[2:(d + 1), :] .*= sqrt(2)
        # even, even moments matrix
        mom = 2 * sqrt(2) ./ [1.0 - i^2 for i in 0:2:2d]
        mom[1] = 2
        Mmom = zeros(d + 1, d + 1)
        f = inv(d * (2d + 1))
        for j in 1:(d + 1), i in 1:(d + 2 - j)
            Mmom[i, j] = mom[i] * mom[j] * f
        end
        Mmom[1, d + 1] *= 0.5
        # cubature weights as matrices on the subgrids
        W = Matrix{Float64}(undef, d + 1, 2d + 1)
        W[:, 1:2:(2d + 1)] .= to2' * Mmom * te1
        W[:, 2:2:(2d + 1)] .= te2' * Mmom * to1
        W[:, [1, (2d + 1)]] .*= 0.5
        W[1, 2:2:(2d + 1)] .*= 0.5
        W[d + 1, 1:2:(2d + 1)] .*= 0.5
        w = vec(W)
    else
        w = Float64[]
    end

    return (U, pts, P0, P0sub, w)
end

function approxfekete_data(n::Int, d::Int, calc_w::Bool)
    @assert d > 0
    @assert n > 1
    (L, U) = get_L_U(n, d)

    # points in the initial interpolation grid
    npts = prod((2d + 1):(2d + n))
    candidate_pts = Matrix{Float64}(undef, npts, n)
    for j in 1:n
        ig = prod((2d + 1 + j):(2d + n))
        cs = cheb2_pts(2d + j)
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
    dom = Box(-ones(n), ones(n))
    (pts, P0, P0sub, w) = make_wsos_arrays(dom, candidate_pts, 2d, U, L, calc_w = calc_w)
    return (U, pts, P0, P0sub, w)
end

# indices of points to keep and quadrature weights at those points
function choose_interp_pts!(
    M::Matrix{Float64},
    candidate_pts::Matrix{Float64},
    deg::Int,
    U::Int,
    calc_w::Bool,
    )

    n = size(candidate_pts, 2)
    u = calc_u(n, deg, candidate_pts)
    m = Vector{Float64}(undef, U)
    m[1] = 2^n
    M[:, 1] .= 1.0

    col = 1
    for t in 1:deg
        for xp in Combinatorics.multiexponents(n, t)
            col += 1
            if any(isodd, xp)
                m[col] = 0.0
            else
                m[col] = m[1] / prod(1.0 - abs2(xp[j]) for j in 1:n)
            end
            @. @views M[:, col] = u[1][:, xp[1] + 1]
            for j in 2:n
                @. @views M[:, col] *= u[j][:, xp[j] + 1]
            end
        end
    end
    F = qr!(Array(M'), Val(true))
    if calc_w
        Qtm = F.Q' * m
        w = UpperTriangular(F.R[:, 1:U]) \ Qtm
    else
        w = Float64[]
    end
    return (F.p[1:U], w)
end

function make_wsos_arrays(
    dom::Domain,
    candidate_pts::Matrix{Float64},
    deg::Int,
    U::Int,
    L::Int;
    calc_w::Bool = false,
    )

    (npts, n) = size(candidate_pts)
    M = Matrix{Float64}(undef, npts, U)
    (keep_pts, w) = choose_interp_pts!(M, candidate_pts, deg, U, calc_w)
    pts = candidate_pts[keep_pts, :]
    P0 = M[keep_pts, 1:L] # subset of polynomial evaluations up to total degree d
    subd = div(deg - get_degree(dom), 2)
    P0sub = view(P0, :, 1:binomial(n + subd, n))

    return (pts, P0, P0sub, w)
end

# fast, sampling-based point selection for general domains
function wsos_sample_params(
    dom::Domain,
    d::Int;
    calc_w::Bool = false,
    sample_factor::Int = 10,
    )
    n = get_dimension(dom)
    (L, U) = get_L_U(n, d)
    candidate_pts = interp_sample(dom, U * sample_factor)
    (pts, P0, P0sub, w) = make_wsos_arrays(dom, candidate_pts, 2d, U, L, calc_w = calc_w)
    g = get_weights(dom, pts)
    PWts = [sqrt.(gi) .* P0sub for gi in g]
    return (U = U, pts = pts, P0 = P0, PWts = PWts, w = w)
end

# TODO should work without sampling too
function get_interp_pts(dom::Domain, deg::Int; sample_factor::Int = 10, calc_w::Bool = false)
    n = get_dimension(dom)
    U = binomial(n + deg, n)
    candidate_pts = interp_sample(dom, U * sample_factor)
    M = Matrix{Float64}(undef, size(candidate_pts, 1), U)
    (keep_pts, w) = choose_interp_pts!(M, candidate_pts, deg, U, calc_w)
    return (candidate_pts[keep_pts, :], w)
end

function recover_lagrange_polys(pts::Matrix{Float64}, deg::Int)
    (U, n) = size(pts)
    DP.@polyvar x[1:n]
    monos = DP.monomials(x, 0:deg)
    vandermonde_inv = inv([monos[j](pts[i, :]) for i in 1:U, j in 1:U])
    lagrange_polys = [dot(vandermonde_inv[:, i], monos) for i in 1:U]
    return lagrange_polys
end

function bilinear_terms(U, pts, P0, PWts, n)
    U_y = div(n * (n + 1), 2)
    ypts = zeros(U_y, n)
    naive_pts = zeros(U * U_y, 2n)
    row = 0
    for i in 1:n, j in i:n
        row += 1
        ypts[row, i] = 1
        ypts[row, j] = 1
        pt_range = ((row - 1) * U + 1):(row * U)
        naive_pts[pt_range, 1:n] = pts
        naive_pts[pt_range, n + i] .= 1
        naive_pts[pt_range, n + j] .= 1
    end
    naive_U = U_y * U
    naive_P0 = kron(ypts, P0)
    if !isempty(PWts)
        naive_PWts = [kron(ypts, pwt) for pwt in PWts]
        return (naive_U, naive_pts, naive_P0, naive_PWts)
    else
        return (naive_U, naive_pts, naive_P0, [])
    end
end

function soc_terms(U, pts, P0, PWts, m)
    U_y = m
    n = size(pts, 2)
    ypts = zeros(U_y, m)
    naive_pts = zeros(U * U_y, n + m)

    naive_pts[1:U, 1:n] = pts
    naive_pts[1:U, n + 1] .= 1

    ypts[1, 1] = 1
    naive_pts[1:U, 1:n] = pts
    naive_pts[1:U, 1] .= 1
    for i in 2:m
        ypts[i, i] = 1
        ypts[i, 1] = 1
        pt_range = ((i - 1) * U + 1):(i * U)
        naive_pts[pt_range, 1:n] = pts
        naive_pts[pt_range, n + i] .= 1
        naive_pts[pt_range, n + 1] .= 1
    end
    naive_U = U_y * U
    naive_P0 = kron(ypts, P0)
    if !isempty(PWts)
        naive_PWts = [kron(ypts, pwt) for pwt in PWts]
        return (naive_U, naive_pts, naive_P0, naive_PWts)
    else
        return (naive_U, naive_pts, naive_P0, [])
    end
end
