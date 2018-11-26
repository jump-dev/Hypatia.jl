#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified/inspired from files in https://github.com/dpapp-github/alfonso/
- https://github.com/dpapp-github/alfonso/blob/master/ChebInterval.m
- https://github.com/dpapp-github/alfonso/blob/master/PaduaSquare.m
- https://github.com/dpapp-github/alfonso/blob/master/FeketeCube.m
and Matlab files in the packages
- Chebfun http://www.chebfun.org/
- Padua2DM by M. Caliari, S. De Marchi, A. Sommariva, and M. Vianello http://profs.sci.univr.it/~caliari/software.htm
=#

# slow but high-quality hyperrectangle/box point selections
function interp_box(n::Int, d::Int; calc_w::Bool=false)
    if n == 1
        return cheb2_data(d, calc_w)
    elseif n == 2
        return padua_data(d, calc_w) # or approxfekete_data(n, d)
    elseif n > 2
        return approxfekete_data(n, d, calc_w)
    end
end

# return k Chebyshev points of the second kind
cheb2_pts(k::Int) = [-cospi(j/(k-1)) for j in 0:k-1]

function calc_u(n::Int, d::Int, pts::Matrix{Float64})
    u = Vector{Matrix{Float64}}(undef, n)
    for j in 1:n
        uj = u[j] = Matrix{Float64}(undef, size(pts, 1), d+1)
        uj[:,1] .= 1.0
        @. @views uj[:,2] = pts[:,j]
        for t in 3:d+1
            @. @views uj[:,t] = 2.0*uj[:,2]*uj[:,t-1] - uj[:,t-2]
        end
    end
    return u
end

function cheb2_data(d::Int, calc_w::Bool)
    @assert d > 1
    L = 1 + d
    U = 1 + 2d

    # Chebyshev points for degree 2d
    pts = reshape(cheb2_pts(U), :, 1)

    # evaluations
    P0 = calc_u(1, d, pts)[1]
    P = Array(qr(P0).Q)

    # weights for Clenshaw-Curtis quadrature at pts
    if calc_w
        wa = Float64[2/(1 - j^2) for j in 0:2:U-1]
        append!(wa, wa[floor(Int, U/2):-1:2])
        w = real.(FFTW.ifft(wa))
        w[1] = w[1]/2
        push!(w, w[1])
    else
        w = Float64[]
    end

    return (L=L, U=U, pts=pts, P0=P0, P=P, w=w)
end

function padua_data(d::Int, calc_w::Bool)
    @assert d > 1
    L = binomial(2+d, 2)
    U = binomial(2+2d, 2)

    # Padua points for degree 2d
    cheba = cheb2_pts(2d+1)
    chebb = cheb2_pts(2d+2)
    pts = Matrix{Float64}(undef, U, 2)
    j = 1
    for a in 0:2d
        for b in 0:2d+1
            if iseven(a+b)
                pts[j,1] = -cheba[a+1]
                pts[U+1-j,2] = -chebb[2d+2-b]
                j += 1
            end
        end
    end

    # evaluations
    u = calc_u(2, d, pts)
    P0 = Matrix{Float64}(undef, U, L)
    P0[:,1] .= 1
    col = 1
    for t in 1:d
        for xp in Combinatorics.multiexponents(2, t)
            col += 1
            P0[:,col] .= u[1][:,xp[1]+1] .* u[2][:,xp[2]+1]
        end
    end
    P = Array(qr(P0).Q)

    # cubature weights at Padua points
    # even-degree Chebyshev polynomials on the subgrids
    if calc_w
        te1 = [cospi(i*j/(2d)) for i in 0:2:2d, j in 0:2:2d]
        to1 = [cospi(i*j/(2d)) for i in 0:2:2d, j in 1:2:2d]
        te2 = [cospi(i*j/(2d+1)) for i in 0:2:2d, j in 0:2:2d+1]
        to2 = [cospi(i*j/(2d+1)) for i in 0:2:2d, j in 1:2:2d+1]
        te1[2:d+1,:] .*= sqrt(2)
        to1[2:d+1,:] .*= sqrt(2)
        te2[2:d+1,:] .*= sqrt(2)
        to2[2:d+1,:] .*= sqrt(2)
        # even, even moments matrix
        mom = 2*sqrt(2)./[1 - i^2 for i in 0:2:2d]
        mom[1] = 2
        Mmom = zeros(d+1, d+1)
        f = 1/(d*(2d+1))
        for j in 1:d+1, i in 1:d+2-j
            Mmom[i,j] = mom[i]*mom[j]*f
        end
        Mmom[1,d+1] /= 2
        # cubature weights as matrices on the subgrids
        W = Matrix{Float64}(undef, d+1, 2d+1)
        W[:,1:2:2d+1] .= to2'*Mmom*te1
        W[:,2:2:2d+1] .= te2'*Mmom*to1
        W[:,[1,2d+1]] ./= 2
        W[1,2:2:2d+1] ./= 2
        W[d+1,1:2:2d+1] ./= 2
        w = vec(W)
    else
        w = Float64[]
    end

    return (L=L, U=U, pts=pts, P0=P0, P=P, w=w)
end

function approxfekete_data(n::Int, d::Int, calc_w::Bool)
    @assert d > 1
    @assert n > 1
    L = binomial(n+d, n)
    U = binomial(n+2d, n)

    # points in the initial interpolation grid
    npts = prod(2d+1:2d+n)
    ipts = Matrix{Float64}(undef, npts, n)
    for j in 1:n
        ig = prod(2d+1+j:2d+n)
        cs = cheb2_pts(2d+j)
        i = 1
        l = 1
        while true
            ipts[i:i+ig-1,j] .= cs[l]
            i += ig
            l += 1
            if l >= 2d+1+j
                if i >= npts
                    break
                end
                l = 1
            end
        end
    end

    # evaluations on the initial interpolation grid
    u = calc_u(n, 2d, ipts)
    m = Vector{Float64}(undef, U)
    m[1] = 2^n
    M = Matrix{Float64}(undef, npts, U)
    M[:,1] .= 1.0

    col = 1
    for t in 1:2d
        for xp in Combinatorics.multiexponents(n, t)
            col += 1
            if any(isodd, xp)
                m[col] = 0.0
            else
                m[col] = m[1]/prod(1.0 - abs2(xp[j]) for j in 1:n)
            end
            @. @views M[:,col] = u[1][:,xp[1]+1]
            for j in 2:n
                @. @views M[:,col] *= u[j][:,xp[j]+1]
            end
        end
    end

    Mp = Array(M')
    F = qr!(Mp, Val(true))
    keep_pnt = F.p[1:U]

    pts = ipts[keep_pnt,:] # subset of points indexed with the support of w
    P0 = M[keep_pnt,1:L] # subset of polynomial evaluations up to total degree d
    P = Array(qr(P0).Q)

    if calc_w
        Qtm = F.Q'*m
        w = UpperTriangular(F.R[:,1:U])\Qtm
    else
        w = Float64[]
    end

    return (L=L, U=U, pts=pts, P0=P0, P=P, w=w)
end


# fast, sampling-based point selection for general domains

# domains
abstract type InterpDomain end

# hyperrectangle/box domain
mutable struct Box <: InterpDomain
    l::Vector{Float64}
    u::Vector{Float64}
    function Box(l::Vector{Float64}, u::Vector{Float64})
        @assert length(l) == length(u)
        d = new()
        d.l = l
        d.u = u
        return d
    end
end

dimension(d::Box) = length(d.l)

function interp_sample(d::Box, npts::Int)
    dim = dimension(d)
    pts = rand(npts, dim) .- 0.5
    shift = (d.u + d.l)/2.0
    for i in 1:npts
        pts[i,:] = pts[i,:] .* (d.u - d.l) + shift
    end
    return pts
end

function get_bss(dom::Box, x)
    bss = SemialgebraicSets.BasicSemialgebraicSet{Float64, DynamicPolynomials.Polynomial{true, Float64}}()
    for i in 1:dimension(dom)
        SemialgebraicSets.addinequality!(bss, (-x[i] + dom.u[i]) * (x[i] - dom.l[i]))
    end
    return bss
end

function get_weights(dom::Box, pts::Matrix{Float64}; count::Int = size(pts, 2))
    @assert count == length(dom.l) == length(dom.u)
    g = [(pts[:,i] .- dom.l[i]) .* (dom.u[i] .- pts[:,i]) for i in 1:count]
    @assert all(all(gi .>= 0.0) for gi in g)
    return g
end

# Euclidean hyperball domain
mutable struct Ball <: InterpDomain
    c::Vector{Float64}
    r::Float64
    function Ball(c::Vector{Float64}, r::Float64)
        d = new()
        d.c = c
        d.r = r
        return d
    end
end

dimension(d::Ball) = length(d.c)

function interp_sample(d::Ball, npts::Int)
    dim = dimension(d)
    pts = randn(npts, dim)
    norms = sum(abs2, pts, dims=2)
    pts .*= d.r ./ sqrt.(norms) # scale
    norms ./= 2.0
    pts .*= sf_gamma_inc_Q.(norms, dim/2) .^ inv(dim) # sf_gamma_inc_Q is the normalized incomplete gamma function
    for i in 1:dim
        pts[:, i] .+= d.c[i] # shift
    end
    return pts
end

get_bss(dom::Ball, x) = SemialgebraicSets.@set(sum((x - dom.c).^2) <= dom.r^2)

function get_weights(dom::Ball, pts::Matrix{Float64}; count::Int = size(pts, 2))
    @assert count == length(dom.c)
    g = [dom.r^2 - sum((pts[j, 1:count] - dom.c).^2) for j in 1:size(pts, 1)]
    @assert all(g .>= 0.0)
    return [g]
end

# hyperellipse domain: (x-c)'Q(x-c) \leq 1
mutable struct Ellipsoid <: InterpDomain
    c::Vector{Float64}
    Q::AbstractMatrix{Float64}
    function Ellipsoid(c::Vector{Float64}, Q::AbstractMatrix{Float64})
        @assert length(c) == size(Q, 1)
        d = new()
        d.c = c
        d.Q = Q
        return d
    end
end

dimension(d::Ellipsoid) = length(d.c)

function interp_sample(d::Ellipsoid, npts::Int)
    dim = dimension(d)
    pts = randn(npts, dim)
    norms = sum(abs2, pts, dims=2)
    for i in 1:npts
        pts[i,:] ./= sqrt(norms[i]) # scale
    end
    norms ./= 2.0
    pts .*= sf_gamma_inc_Q.(norms, dim/2) .^ inv(dim) # sf_gamma_inc_Q is the normalized incomplete gamma function

    # TODO rewrite this part
    rotscale = cholesky(d.Q).U
    # fchol = cholesky(inv(d.Q)) # TODO this is unnecessarily expensive and prone to numerical issues: take cholesky of Q then divide by it later
    for i in 1:npts
        # @assert norm(pts[i,:]) < 1.0
        pts[i,:] = rotscale\pts[i,:] # rotate/scale
        # pts[i,:] = fchol.L * pts[i,:] # rotate/scale
    end


    for i in 1:dim
        pts[:, i] .+= d.c[i] # shift
    end
    return pts
end

get_bss(dom::Ellipsoid, x) = SemialgebraicSets.@set((x - dom.c)' * dom.Q * (x - dom.c) <= 1.0)

function get_weights(dom::Ellipsoid, pts::Matrix{Float64}; count::Int = size(pts, 2))
    @assert count == length(dom.c)
    g = [1.0 - (pts[j, 1:count] - dom.c)' * dom.Q * (pts[j, 1:count] - dom.c) for j in 1:size(pts, 1)]
    @assert all(g .>= 0.0)
    return [g]
end



# TODO refactor common code here and in approxfekete_data
function get_large_P(ipts::Matrix{Float64}, d::Int, U::Int)
    (npts, n) = size(ipts)
    u = calc_u(n, 2d, ipts)
    m = Vector{Float64}(undef, U)
    m[1] = 2^n
    M = Matrix{Float64}(undef, npts, U)
    M[:,1] .= 1.0

    col = 1
    for t in 1:2d
        for xp in Combinatorics.multiexponents(n, t)
            col += 1
            if any(isodd, xp)
                m[col] = 0.0
            else
                m[col] = m[1]/prod(1.0 - abs2(xp[j]) for j in 1:n)
            end
            @. @views M[:,col] = u[1][:,xp[1]+1]
            for j in 2:n
                @. @views M[:,col] *= u[j][:,xp[j]+1]
            end
        end
    end

    return (M, m)
end

function interp_sample(
    dom::InterpDomain,
    n::Int,
    d::Int;
    calc_w::Bool = false,
    pts_factor::Int = n,
    ortho_wts::Bool = true,
    )

    L = binomial(n+d,n)
    U = binomial(n+2d, n)
    candidate_pts = interp_sample(dom, U * pts_factor)
    (M, m) = get_large_P(candidate_pts, d, U)
    Mp = Array(M')
    F = qr!(Mp, Val(true))
    keep_pnt = F.p[1:U]
    pts = candidate_pts[keep_pnt,:] # subset of points indexed with the support of w
    P0 = M[keep_pnt, 1:L] # subset of polynomial evaluations up to total degree d
    P = Array(qr(P0).Q)
    P0sub = view(P0, :, 1:binomial(n+d-1, n))

    g = Hypatia.get_weights(dom, pts)
    PWts = [sqrt.(gi) .* P0sub for gi in g]
    if ortho_wts
        PWts = [Array(qr!(W).Q) for W in PWts] # orthonormalize
    end

    if calc_w
        Qtm = F.Q'*m
        w = UpperTriangular(F.R[:,1:U])\Qtm
    else
        w = Float64[]
    end

    return (L=L, U=U, pts=pts, P0=P0, P=P, PWts=PWts, w=w)
end
