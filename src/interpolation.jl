#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified/inspired from files in https://github.com/dpapp-github/alfonso/
- https://github.com/dpapp-github/alfonso/blob/master/ChebInterval.m
- https://github.com/dpapp-github/alfonso/blob/master/PaduaSquare.m
- https://github.com/dpapp-github/alfonso/blob/master/FeketeCube.m
and Matlab files in the packages
- Chebfun http://www.chebfun.org/
- Padua2DM by M. Caliari, S. De Marchi, A. Sommariva, and M. Vianello http://profs.sci.univr.it/~caliari/software.htm
=#

function interpolate(n::Int, d::Int; sample::Bool = true, sample_factor::Int = 50, calc_w::Bool = false)
    if n == 1
        return cheb2_data(d, calc_w)
    elseif n == 2
        return padua_data(d, calc_w) # or approxfekete_data(n, d)
    elseif n > 2
        return approxfekete_data(n, d, sample, sample_factor, calc_w)
    end
end

# return k Chebyshev points of the second kind
cheb2_pts(k::Int) = [-cospi(j/(k-1)) for j in 0:k-1]

function get_LU(n::Int, d::Int)
    L = binomial(n + d, n)
    U = binomial(n + 2d, n)
    return (L, U)
end

function calc_u(n::Int, d::Int, pts::Matrix{Float64})
    @assert d > 0
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
    @assert d > 0
    U = 2d + 1

    # Chebyshev points for degree 2d
    pts = reshape(cheb2_pts(U), :, 1)

    # evaluations
    P0 = calc_u(1, d, pts)[1]

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

    return (U, pts, P0, w)
end

function padua_data(d::Int, calc_w::Bool)
    @assert d > 0
    (L, U) = get_LU(2, d)

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

    return (U, pts, P0, w)
end

function approxfekete_data(n::Int, d::Int, sample::Bool, sample_factor::Int, calc_w::Bool)
    @assert d > 0
    @assert n > 1
    (L, U) = get_LU(n, d)
    @show L, U

    if sample
        candidate_pts = 2.0 * (rand(sample_factor * U, n) .- 0.5)
    else
        # points in the initial interpolation grid
        num_pts = prod(2d+1:2d+n)
        candidate_pts = Matrix{Float64}(undef, num_pts, n)
        for j in 1:n
            ig = prod(2d+1+j:2d+n)
            cs = cheb2_pts(2d+j)
            i = 1
            l = 1
            while true
                candidate_pts[i:i+ig-1,j] .= cs[l]
                i += ig
                l += 1
                if l >= 2d+1+j
                    if i >= num_pts
                        break
                    end
                    l = 1
                end
            end
        end
    end
    (pts, P0, w) = select_points(candidate_pts, d, U, L, calc_w)
    return (U, pts, P0, w)
end

function select_points(
    candidate_pts::Matrix{Float64},
    d::Int,
    U::Int,
    L::Int,
    calc_w::Bool,
    )
    (num_pts, n) = size(candidate_pts)
    u = calc_u(n, 2d, candidate_pts)
    m = Vector{Float64}(undef, U)
    m[1] = 2^n
    M = Matrix{Float64}(undef, num_pts, U)
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

    F = qr!(Array(M'), Val(true))
    keep_pnt = F.p[1:U]
    pts = candidate_pts[keep_pnt,:] # subset of points indexed with the support of w
    P0 = M[keep_pnt, 1:L] # subset of polynomial evaluations up to total degree d

    if calc_w
        Qtm = F.Q'*m
        w = UpperTriangular(F.R[:,1:U])\Qtm
    else
        w = Float64[]
    end

    return (pts, P0, w)
end

# TODO decide what kind of object this should be, or if it should be a SemialgebraicSets semialgebraicset
struct SemialgebraicSet
    poly::Function
    idxs::Vector{Int}
    degree::Int
    function SemialgebraicSet(poly::Function, idxs::Vector{Int}, degree::Int)
        @assert mod(degree, 2) == 0 # TODO decide what to do otherwise
        return new(poly, idxs, degree)
    end
end

struct Domain
    sets::Vector{SemialgebraicSet}
end

function build_weights(pts::Matrix{Float64}, dom::Domain, d::Int)
    (U, n) = size(pts)
    lower_dims = Vector{Int}(undef, length(dom.sets))
    weight_vecs = [Vector{Float64}(undef, U) for _ in 1:length(dom.sets)]
    for i in eachindex(dom.sets)
        subd = d - div(dom.sets[i].degree, 2)
        lower_dims[i] = binomial(n + subd, n)
        weight_vecs[i] = [dom.sets[i].poly(pts[j, dom.sets[i].idxs]) for j in 1:U]
    end
    return (weight_vecs, lower_dims)
end

# TODO assumes ball/box/ellipsoid means in all dimensions

function Box(l::Vector{Float64}, u::Vector{Float64}, idxs::Vector{Int})
    @assert length(l) == length(u) == length(idxs)
    n = length(idxs)
    dom = Domain(Vector{SemialgebraicSet}(undef, n + 1))
    dom.sets[1] = SemialgebraicSet(x -> 1.0, idxs, 0) # TODO remove
    for i in 2:n+1
        f(x) = (@assert length(x) == 1; (x[1] - l[i - 1]) * (u[i - 1] - x[1]))
        dom.sets[i] = SemialgebraicSet(f, [i - 1], 2)
    end
    return dom
end

function Ball(c::Vector{Float64}, r::Float64, idxs::Vector{Int})
    w1 = SemialgebraicSet(x -> 1.0, idxs, 0) # TODO remove
    f(x) = r^2 - sum(abs2, (x - c)) # TODO check dims
    return Domain([w1; SemialgebraicSet(f, idxs, 2)])
end

function Ellipsoid(c::Vector{Float64}, Q::AbstractMatrix{Float64}, idxs::Vector{Int})
    w1 = SemialgebraicSet(x -> 1.0, idxs, 0) # TODO remove
    f(x) = 1.0 - (x - c)' * Q * (x - c) # TODO check dims
    return Domain([w1; SemialgebraicSet(f, idxs, 2)])
end

function FreeDomain(idxs::Vector{Int})
    return Domain(SemialgebraicSet(x -> 1.0, idxs, 0))
end

function interpolate(dom::Domain, n::Int, d::Int; sample::Bool = true, sample_factor::Int = 10, calc_w::Bool = false)
    (U, pts, P0, w) = interpolate(n, d, sample = sample, sample_factor = sample_factor, calc_w = calc_w)
    (weight_vecs, lower_dims) = build_weights(pts, dom, d)
    return (U, pts, P0, weight_vecs, lower_dims, w)
end
