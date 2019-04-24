#=
Copyright 2018, Chris Coey and contributors

(closure of) hypograph of perspective of (natural) log of determinant of a (row-wise lower triangle i.e. svec space) symmetric positive define matrix
(smat space) (u in R, v in R_+, w in S_+) : u <= v*logdet(W/v)
(see equivalent MathOptInterface LogDetConeConeTriangle definition)

barrier (guessed, based on analogy to hypoperlog barrier)
-log(v*logdet(W/v) - u) - logdet(W) - log(v)

TODO only use one decomposition on Symmetric(W) for isposdef and logdet
TODO symbolically calculate gradient and Hessian

TODO symmetrize
=#

mutable struct HypoPerLogdet <: Cone
    use_dual::Bool
    dim::Int
    side::Int
    point::AbstractVector{Float64}
    mat::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function HypoPerLogdet(dim::Int, is_dual::Bool)
        cone = new()
        cone.use_dual = is_dual
        cone.dim = dim
        side = round(Int, sqrt(dim - 2))
        cone.side = side
        cone.mat = Matrix{Float64}(undef, side, side)
        cone.g = Vector{Float64}(undef, dim)
        cone.H = Matrix{Float64}(undef, dim, dim)
        cone.H2 = similar(cone.H)
        function barfun(point)
            u = point[1]
            v = point[2]
            Wvec = view(point, 3:dim)
            W = reshape(Wvec, side, side)
            return -log(v * logdet(W / v) - u) - logdet(W) - log(v)
        end
        cone.barfun = barfun
        cone.diffres = DiffResults.HessianResult(cone.g)
        return cone
    end
end

HypoPerLogdet(dim::Int) = HypoPerLogdet(dim, false)

get_nu(cone::HypoPerLogdet) = cone.side + 2

function set_initial_point(arr::AbstractVector{Float64}, cone::HypoPerLogdet)
    arr[1] = -1.0
    arr[2] = 1.0
    arr[3:end] .= vec(Matrix(1.0I, cone.side, cone.side))
    # smat_to_svec!(view(arr, 3:cone.dim), Matrix(1.0I, cone.side, cone.side))
    return arr
end

function check_in_cone(cone::HypoPerLogdet)
    u = cone.point[1]
    v = cone.point[2]
    W = reshape(cone.point[3:end], cone.side, cone.side)
    if v <= 0.0 || !isposdef(Symmetric(W)) || u >= v * logdet(Symmetric(W) / v) # TODO only use one decomposition on Symmetric(W) for isposdef and logdet
        return false
    end

    L = logdet(W / v)
    z = v * L - u
    n = cone.side
    Wi = inv(W)

    gu = 1 / z
    gv = (n - L) / z - 1 / v
    gwmat = -v / z * Wi - Wi
    gw = vec(gwmat)

    Huu = 1 / z / z
    Huv = (n - L) / z / z
    Huw = -(v * Wi) / z / z

    # Hvv = (-n + L) * (-n * v + L) / z / z + n / z + 1 / v^2 # TODO figure out why this is wrong...
    Hvv = (-n + L)^2 / z / z + n / (v * z) + 1 / v^2
    Hvw = (-n + L) * v * Wi / z / z - Wi / z

    Hww = zeros(n^2, n^2)
    vzi = v / z
    fact = vzi * (1 + vzi)
    k = 0
    for j in 1:n, i in 1:n
        k += 1
        k2 = 0
        for j2 in 1:n, i2 in 1:n
            k2 += 1
            Hww[k, k2] = Wi[i2, j] * Wi[i, j2] * v / z + Wi[i, j] * Wi[i2, j2] * v^2 / z^2 +  Wi[i2, j] * Wi[i, j2]
        end
    end
    # Hww *= (1 + fact)

    g = [gu, gv, gw...]


    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.g .= DiffResults.gradient(cone.diffres)
    cone.H .= DiffResults.hessian(cone.diffres)

    # @show size(cone.g), size(g)
    @show cone.H[(2 + n^2):end, (2 + n^2):end] ./ vec(Hww)

    return factorize_hess(cone)
end
