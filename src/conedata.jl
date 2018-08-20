
abstract type PrimitiveCone end

# TODO reorder primitive cones so easiest ones to check incone are first
mutable struct Cone
    prms::Vector{PrimitiveCone}
    idxs::Vector{AbstractUnitRange}
end

# TODO can parallelize the functions acting on Cone
function load_tx(cone::Cone, tx::Vector{Float64})
    for k in eachindex(cone.prms)
        cone.prms[k].point .= tx[cone.idxs[k]]
    end
    return nothing
end

incone(cone::Cone) = all(incone_prm, cone.prms)

function get_g!(g::Vector{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        g[cone.idxs[k]] .= calcg_prm(cone.prms[k])
    end
    return g
end

function get_Hi_mat_t!(Hi_mat::AbstractMatrix{Float64}, mat::AbstractMatrix{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        Hi_mat[cone.idxs[k],:] .= calcHiprod_prm(cone.prms[k], mat[:,cone.idxs[k]]') # TODO maybe faster with views or by saving submatrix of A in cone object
    end
    return Hi_mat
end

function get_Hi_vec!(Hi_vec::Vector{Float64}, vec::Vector{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        Hi_vec[cone.idxs[k]] .= calcHiprod_prm(cone.prms[k], vec)
    end
    return Hi_vec
end


#=
predefined standard primitive cone types
=#

# nonnegative orthant cone
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


# polynomial (weighted) sum of squares cone (parametrized by ip and ipwt)
mutable struct SumOfSqrData <: ConeData
    dim::Int
    ip::Matrix{Float64}
    ipwt::Vector{Matrix{Float64}}
    pnt::Vector{Float64}
    Hfact
    hasgH::Bool
    hasHi::Bool
    grad::Vector{Float64}
    Hi::Matrix{Float64}

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
        k.Hi = inv(k.Hfact) # TODO maybe should do L\L'\A (see math in original code)
        k.hasHi = true
    end
    return k.Hi
end
