#=
Copyright 2018, Chris Coey and contributors

(closure of) hypograph of perspective of (natural) log of determinant of a (row-wise lower triangle i.e. svec space) symmetric positive define matrix
(smat space) (u in R, v in R_+, w in S_+) : u <= v*logdet(W/v)
(see equivalent MathOptInterface LogDetConeConeTriangle definition)

barrier (guessed, based on analogy to hypoperlog barrier)
-log(v*logdet(W/v) - u) - logdet(W) - log(v)
=#

mutable struct HypoPerLogdet{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int

    point::AbstractVector{Float64}
    mat::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F

    function HypoPerLogdet(dim::Int, is_dual::Bool)
        cone = new()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.side = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
        return cone
    end
end

HypoPerLogdet(dim::Int) = HypoPerLogdet(dim, false)

function setup_data(cone::HypoPerLogdet)
    dim = cone.dim
    side = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
    side = cone.side
    cone.mat = Matrix{Float64}(undef, side, side)
    cone.g = Vector{Float64}(undef, dim)
    cone.H = Matrix{Float64}(undef, dim, dim)
    cone.H2 = similar(cone.H)
    return
end

get_nu(cone::HypoPerLogdet) = cone.side + 2

function set_initial_point(arr::AbstractVector{Float64}, cone::HypoPerLogdet)
    arr[1] = -1.0
    arr[2] = 1.0
    smat_to_svec!(view(arr, 3:cone.dim), Matrix(1.0I, cone.side, cone.side))
    return arr
end

function check_in_cone(cone::HypoPerLogdet)
    u = cone.point[1]
    v = cone.point[2]
    if v <= 0.0
        return false
    end
    W = cone.mat
    svec_to_smat!(W, view(cone.point, 3:cone.dim))
    F = cholesky(Symmetric(W), Val(true), check = false)
    if !isposdef(F) || u >= v * (logdet(F) - cone.side * log(v))
        return false
    end

    L = logdet(W / v)
    z = v * L - u
    Wi = inv(W)
    n = cone.side
    dim = cone.dim
    vzi = v / z

    cone.g[1] = 1 / z
    cone.g[2] = (n - L) / z - 1 / v
    gwmat = -Wi * (1 + vzi)
    smat_to_svec!(view(cone.g, 3:dim), gwmat)

    cone.H[1, 1] = 1 / z / z
    cone.H[1, 2] = (n - L) / z / z
    Huwmat = -vzi * Wi / z
    smat_to_svec!(view(cone.H, 1, 3:dim), Huwmat)

    cone.H[2, 2] = (-n + L)^2 / z / z + n / (v * z) + 1 / v / v
    Hvwmat = ((-n + L) * vzi - 1) * Wi / z
    smat_to_svec!(view(cone.H, 2, 3:dim), Hvwmat)

    k = 3
    for i in 1:n, j in 1:i
        k2 = 3
        for i2 in 1:n, j2 in 1:i2
            if (i == j) && (i2 == j2)
                cone.H[k2, k] = abs2(Wi[i2, i]) * (vzi + 1) + Wi[i, i] * Wi[i2, i2] * vzi^2
            elseif (i != j) && (i2 != j2)
                cone.H[k2, k] = (Wi[i2, i] * Wi[j, j2] + Wi[j2, i] * Wi[j, i2]) * (vzi + 1) + 2 * Wi[i, j] * Wi[i2, j2] * vzi^2
            else
                cone.H[k2, k] = rt2 * (Wi[i2, i] * Wi[j, j2] * (vzi + 1) + Wi[i, j] * Wi[i2, j2] * vzi^2)
            end
            if k2 == k
                break
            end
            k2 += 1
        end
        k += 1
    end

    # @assert isapprox(Symmetric(cone.H, :U) * cone.point, -cone.g, atol = 1e-6, rtol = 1e-6)

    return factorize_hess(cone)
end
