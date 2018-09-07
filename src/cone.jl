
#=
cone object
=#

abstract type PrimitiveCone end

# TODO reorder primitive cones so easiest ones to check incone are first
mutable struct Cone
    prms::Vector{PrimitiveCone}
    idxs::Vector{AbstractVector{Int}}

    function Cone(prms::Vector{PrimitiveCone}, idxs::Vector{AbstractVector{Int}})
        @assert length(prms) == length(idxs)
        for k in eachindex(prms)
            @assert dimension(prms[k]) == length(idxs[k])
        end
        cone = new()
        cone.prms = prms
        cone.idxs = idxs
        return cone
    end
end
Cone() = Cone(PrimitiveCone[], AbstractVector{Int}[])

function addprimitivecone!(cone, prm, idx)
    @assert dimension(prm) == length(idx)
    push!(cone.prms, prm)
    push!(cone.idxs, idx)
    return cone
end

# calculate complexity parameter of the barrier (sum of the primitive cone barrier parameters)
barrierpar(cone::Cone) = sum(barrierpar_prm(prm) for prm in cone.prms)

function getintdir!(arr::Vector{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        getintdir_prm!(view(arr, cone.idxs[k]), cone.prms[k])
    end
    return arr
end

# TODO can parallelize the functions acting on Cone
function loadpnt!(cone::Cone, pnt::Vector{Float64})
    for k in eachindex(cone.prms)
        loadpnt_prm!(cone.prms[k], view(pnt, cone.idxs[k]))
    end
    return nothing
end

incone(cone::Cone) = all(incone_prm, cone.prms)

function calcg!(g::Vector{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcg_prm!(view(g, cone.idxs[k]), cone.prms[k])
    end
    return g
end

function calcHiarr!(Hi_mat::AbstractMatrix{Float64}, mat::AbstractMatrix{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcHiarr_prm!(view(Hi_mat, cone.idxs[k], :), view(mat, cone.idxs[k], :), cone.prms[k])
    end
    return Hi_mat
end

function calcHiarr!(Hi_vec::AbstractVector{Float64}, vec::AbstractVector{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcHiarr_prm!(view(Hi_vec, cone.idxs[k]), view(vec, cone.idxs[k]), cone.prms[k])
    end
    return Hi_vec
end

# utilities for converting between smat and svec forms (no rescaling) for symmetric matrices
# TODO only need to do upper triangle if use symmetric matrix types
const rt2 = sqrt(2)
const rt2i = inv(rt2)

function mattovec!(vec::AbstractVector{Float64}, mat::AbstractMatrix{Float64})
    k = 1
    (m, n) = size(mat)
    for i in 1:m, j in i:n
        if i == j
            vec[k] = mat[i,j]
        else
            vec[k] = rt2*mat[i,j]
        end
        k += 1
    end
    return vec
end

function vectomat!(mat::AbstractMatrix{Float64}, vec::AbstractVector{Float64})
    k = 1
    (m, n) = size(mat)
    for i in 1:m, j in i:n
        if i == j
            mat[i,j] = vec[k]
        else
            mat[i,j] = mat[j,i] = rt2i*vec[k]
        end
        k += 1
    end
    return mat
end
