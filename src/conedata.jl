
# export ConeData, NonnegData, SumOfSqrData

abstract type ConeData end

#=
 Nonnegative cone
=#
#
# mutable struct NonnegData <: ConeData
#     dim::Int
#     pnt::Vector{Float64}
#     has_g::Bool
#     g::Vector{Float64}
#     g_prev::Vector{Float64}
#     has_Hi::Bool
#     Hi::Diagonal{Float64}
#     Hi_prev::Diagonal{Float64}
#
#     function NonnegData(dim)
#         k = new()
#         k.dim = dim
#         k.has_g = false
#         k.g = Vector{Float64}(undef, dim)
#         k.g_prev = copy(k.g)
#         k.has_Hi = false
#         k.Hi = Diagonal{Float64}(copy(k.g))
#         k.Hi_prev = copy(k.Hi)
#         return k
#     end
# end
#
# dimension(k::NonnegData) = k.dim
#
# barpar(k::NonnegData) = k.dim
#
# function load_txk(k::NonnegData, pnt::Vector{Float64}, save_prev::Bool)
#     @assert length(pnt) == k.dim
#     if save_prev # may want to use previously calculated values, so store
#         k.g_prev .= k.g
#         k.Hi_prev.diag .= k.Hi.diag
#     end
#     k.pnt = pnt
#     k.has_g = false
#     k.has_Hi = false
#     return nothing
# end
#
# function use_prevk(k::NonnegData)
#     k.g .= k.g_prev
#     k.Hi.diag .= k.Hi_prev.diag
#     return nothing
# end
#
# inconek(k::NonnegData) = all(x -> (x > 0.0), k.pnt)
#
# function calc_gk(k::NonnegData)
#     if !k.has_g
#         k.g .= -inv.(k.pnt)
#         k.has_g = true
#     end
#     return k.g
# end
#
# function calc_Hik(k::NonnegData)
#     if !k.has_Hi
#         k.Hi.diag .= abs2.(k.pnt)
#         k.has_Hi = true
#     end
#     return k.Hi
# end
#
# calc_Lk(k::NonnegData) = Diagonal(-k.g)
#

mutable struct NonnegData <: ConeData
    dim::Int
    pnt::Vector{Float64}
    pnt_prev::Vector{Float64}

    function NonnegData(dim)
        k = new()
        k.dim = dim
        k.pnt = Vector{Float64}(undef, dim)
        k.pnt_prev = copy(k.pnt)
        # TODO prealloc etc
        return k
    end
end

dimension(k::NonnegData) = k.dim

barpar(k::NonnegData) = k.dim

function load_txk(k::NonnegData, pnt, save_prev)
    @assert length(pnt) == k.dim
    if save_prev
        k.pnt_prev .= k.pnt
    end
    k.pnt .= pnt
end

use_prevk(k::NonnegData) = (k.pnt .= k.pnt_prev)

inconek(k::NonnegData) = all(x -> (x > 0.0), k.pnt)

calc_gk(k::NonnegData) = -inv.(k.pnt)

calc_Hik(k::NonnegData) = Diagonal(abs2.(k.pnt))

calc_Lk(k::NonnegData) = Diagonal(inv.(k.pnt)) # TODO just -g

#=
 Sum of squares cone
=#
mutable struct SumOfSqrData <: ConeData
    dim::Int
    ip::Matrix{Float64}
    ipwt::Vector{Matrix{Float64}}
    pnt::Vector{Float64}
    Hfact
    hasgH::Bool
    hasHi::Bool
    hasL::Bool
    grad::Vector{Float64}
    grad_prev::Vector{Float64}
    Hi::Matrix{Float64}
    Hi_prev::Matrix{Float64}
    L::Matrix{Float64}
    L_prev::Matrix{Float64}

    function SumOfSqrData(dim, ip, ipwt)
        k = new()
        k.dim = dim
        k.ip = ip
        k.ipwt = ipwt
        k.pnt = Vector{Float64}(undef, dim)
        # TODO prealloc etc
        return k
    end
end

dimension(k::SumOfSqrData) = k.dim

barpar(k::SumOfSqrData) = (size(k.ip, 2) + sum(size(k.ipwt[j], 2) for j in 1:length(k.ipwt)))

function load_txk(k::SumOfSqrData, pnt::Vector{Float64}, save_prev::Bool)
    @assert length(pnt) == k.dim
    if save_prev
        # may want to use previously calculated values, so store
        @assert k.hasgH
        k.grad_prev = calc_gk(k)
        k.Hi_prev = calc_Hik(k)
        k.L_prev = calc_Lk(k)
    end
    k.pnt .= pnt
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

function calc_Hik(k::SumOfSqrData)
    @assert k.hasgH
    if !k.hasHi
        k.Hi = inv(k.Hfact) # TODO maybe should do L\L'\A
        k.hasHi = true
    end
    return k.Hi
end

function calc_Lk(k::SumOfSqrData)
    @assert k.hasgH
    if !k.hasL
        k.L = k.Hfact.L
        k.hasL = true
    end
    return k.L
end
