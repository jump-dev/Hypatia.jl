
abstract type PrimitiveCone end


mutable struct Cone
    primitives::Vector{PrimitiveCone}

end







#=
 Nonnegative cone
=#
#
# mutable struct NonnegData <: ConeData
#     dim::Int
#     pnt::Vector{Float64}
#     has_g::Bool
#     g::Vector{Float64}
#     has_Hi::Bool
#     Hi::Diagonal{Float64}
#
#     function NonnegData(dim)
#         k = new()
#         k.dim = dim
#         k.has_g = false
#         k.g = Vector{Float64}(undef, dim)
#         k.has_Hi = false
#         k.Hi = Diagonal{Float64}(copy(k.g))
#         return k
#     end
# end
#
# dimension(k::NonnegData) = k.dim
#
# barpar(k::NonnegData) = k.dim
#
# function load_txk(k::NonnegData, pnt::Vector{Float64})
#     @assert length(pnt) == k.dim
#     k.pnt = pnt
#     k.has_g = false
#     k.has_Hi = false
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

    function NonnegData(dim)
        k = new()
        k.dim = dim
        k.pnt = Vector{Float64}(undef, dim)
        # TODO prealloc etc
        return k
    end
end

dimension(k::NonnegData) = k.dim

barpar(k::NonnegData) = k.dim

function load_txk(k::NonnegData, pnt)
    @assert length(pnt) == k.dim
    k.pnt .= pnt
end

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
    Hi::Matrix{Float64}
    L::Matrix{Float64}

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

function load_txk(k::SumOfSqrData, pnt::Vector{Float64})
    @assert length(pnt) == k.dim
    k.pnt .= pnt
    k.hasgH = false
    k.hasHi = false
    k.hasL = false
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




# # create cone object functions related to primal cone barrier
# function load_tx!(cone, tx)
#     for k in eachindex(cone)
#         load_txk(cone[k], tx[coneidxs[k]])
#     end
#     return nothing
# end
#
# function check_incone(cone)
#     for k in eachindex(cone)
#         if !inconek(cone[k])
#             return false
#         end
#     end
#     return true
# end
#
# function calc_g!(cone, g)
#     for k in eachindex(cone)
#         g[coneidxs[k]] .= calc_gk(cone[k])
#     end
#     return g
# end
#
# function calc_Hinv_vec!(cone, Hi_vec, v)
#     for k in eachindex(cone)
#         Hi_vec[coneidxs[k]] .= calc_Hik(cone[k])*v[coneidxs[k]]
#     end
#     return Hi_vec
# end
#
# # TODO could save At submatrix inside cone objects
# function calc_Hinv_At!(cone, Hi_At, A)
#     for k in eachindex(cone)
#         Hi_At[coneidxs[k],:] .= calc_Hik(cone[k])*A[:,coneidxs[k]]'
#     end
#     return Hi_At
# end
#
# function calc_nbhd(cone, ts, mu, tk)
#     # sqrt(sum(abs2, L\(ts + mu*g)) + (tau*kap - mu)^2)/mu
#     sumsqr = (tk - mu)^2
#     for k in eachindex(cone)
#         sumsqr += sum(abs2, calc_Lk(cone[k])\(ts[coneidxs[k]] + mu*calc_gk(cone[k])))
#     end
#     return sqrt(sumsqr)/mu
# end
