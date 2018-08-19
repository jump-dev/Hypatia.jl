
# export ConeData, NonnegData, SumOfSqrData

abstract type ConeData end

#=
 Nonnegative cone
=#
mutable struct NonnegData <: ConeData
    dim::Int
    pnt::Vector{Float64}
    hasg::Bool
    g::Vector{Float64}
    g_prev::Vector{Float64}

    function NonnegData(dim)
        k = new()
        k.dim = dim
        k.hasg = false
        k.g = Vector{Float64}(undef, dim)
        k.g_prev = Vector{Float64}(undef, dim)
        return k
    end
end

dimension(k::NonnegData) = k.dim

barpar(k::NonnegData) = k.dim

function load_txk(k::NonnegData, pnt::Vector{Float64}, save_prev::Bool)
    @assert length(pnt) == k.dim
    if save_prev # may want to use previously calculated values, so store
        @assert k.hasg
        k.g_prev .= k.g
    end
    k.pnt = pnt
    k.hasg = false
    return nothing
end

function use_prevk(k::NonnegData)
    k.g .= k.g_prev
    k.hasg = true
    return nothing
end

inconek(k::NonnegData) = all(x -> (x > 0.0), k.pnt)

function calc_gk(k::NonnegData)
    if !k.hasg
        k.g .= -inv.(k.pnt)
        k.hasg = true
    end
    return k.g
end

calc_Wik(k::NonnegData) = Diagonal(k.pnt)

#=
 Sum of squares cone
=#
mutable struct SumOfSqrData <: ConeData
    dim::Int
    ip
    ipwt

    isnewpnt
    pnt

    grad
    Hfact
    hasgH
    hasHi
    Hi
    hasL
    L

    grad_prev
    Hi_prev
    L_prev

    # TODO prealloc etc

    function SumOfSqrData(dim, ip, ipwt)
        k = new()
        k.dim = dim
        k.ip = ip
        k.ipwt = ipwt
        k.isnewpnt = false
        return k
    end
end

dimension(k::SumOfSqrData) = k.dim

barpar(k::SumOfSqrData) = (size(k.ip, 2) + sum(size(k.ipwt[j], 2) for j in 1:length(k.ipwt)))

function load_txk(k::SumOfSqrData, pnt, save_prev)
    @assert length(pnt) == k.dim

    if save_prev
        # may want to use previously calculated values, so store
        @assert k.hasgH
        k.grad_prev = calc_gk(k)
        k.Hi_prev = calc_Hinvk(k)
        k.L_prev = calc_HCholLk(k)
    end

    k.pnt = pnt
    k.isnewpnt = true
    k.hasgH = false
    k.hasHi = false
    k.hasL = false
end

function use_prevk(k::SumOfSqrData)
    k.grad = copy(k.grad_prev) # TODO prealloc not copy
    k.Hi = copy(k.Hi_prev)
    k.L = copy(k.L_prev)

    k.hasgH = true
    k.hasHi = true
    k.hasL = true
end

function inconek(k::SumOfSqrData)
    @assert k.isnewpnt
    k.isnewpnt = false

    F = cholesky!(Symmetric(k.ip'*Diagonal(k.pnt)*k.ip), check=false) # TODO do inplace. TODO could this cholesky of P'DP be faster?
    if !issuccess(F)
        return false
    end
    Vp = F.L\k.ip' # TODO prealloc
    VtVp = Vp'*Vp
    g = -diag(VtVp)
    H = VtVp.^2

    for j in 1:length(k.ipwt)
        F = cholesky!(Symmetric(k.ipwt[j]'*Diagonal(k.pnt)*k.ipwt[j]), check=false)
        if !issuccess(F)
            return false
        end
        Vp = F.L\k.ipwt[j]'
        VtVp = Vp'*Vp
        g .-= diag(VtVp)
        H .+= VtVp.^2
    end

    F = cholesky!(H, check=false)
    if !issuccess(F)
        return false
    end

    k.grad = g
    k.Hfact = F
    k.hasgH = true
    return true
end

function calc_gk(k::SumOfSqrData)
    @assert k.hasgH
    return k.grad
end

function calc_Hinvk(k::SumOfSqrData)
    @assert k.hasgH
    if !k.hasHi
        k.Hi = inv(k.Hfact)
        k.hasHi = true
    end
    return k.Hi
end

function calc_HCholLk(k::SumOfSqrData)
    @assert k.hasgH
    if !k.hasL
        k.L = k.Hfact.L
        k.hasL = true
    end
    return k.L
end
