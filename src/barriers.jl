
export ConeData, PolyNonnegData

abstract type ConeData end

mutable struct PolyNonnegData <: ConeData
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

    function PolyNonnegData(dim, ip, ipwt)
        k = new()
        k.dim = dim
        k.ip = ip
        k.ipwt = ipwt
        k.isnewpnt = false
        return k
    end
end

dimension(k::PolyNonnegData) = k.dim
barpar(k::PolyNonnegData) = k.dim

function load_txk(k::PolyNonnegData, pnt, save_prev)
    @assert length(pnt) == k.dim

    if save_prev
        # may want to use previously calculated values, so store
        @assert k.hasgH
        k.grad_prev = calc_gk(k)
        k.Hi_prev = calc_Hik(k)
        k.L_prev = calc_Lk(k)
    end

    k.pnt = pnt
    k.isnewpnt = true
    k.hasgH = false
    k.hasHi = false
    k.hasL = false
end

function use_prevk(k::PolyNonnegData)
    k.grad = copy(k.grad_prev) # TODO prealloc not copy
    k.Hi = copy(k.Hi_prev)
    k.L = copy(k.L_prev)

    k.hasgH = true
    k.hasHi = true
    k.hasL = true
end

function inconek(k::PolyNonnegData)
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

function calc_gk(k::PolyNonnegData)
    @assert k.hasgH
    return k.grad
end

function calc_Hik(k::PolyNonnegData)
    @assert k.hasgH
    if !k.hasHi
        k.Hi = inv(k.Hfact)
        k.hasHi = true
    end
    return k.Hi
end

function calc_Lk(k::PolyNonnegData)
    @assert k.hasgH
    if !k.hasL
        k.L = k.Hfact.L
        k.hasL = true
    end
    return k.L
end





# get barrier function parameter
# barpar(k::NonnegativesData) = k.dimension
# barpar(k::SecondOrderConeData) = 2
