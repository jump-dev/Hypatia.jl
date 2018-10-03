#=
Copyright 2018, Chris Coey and contributors
=#

# cone object
abstract type PrimitiveCone end

# TODO reorder primitive cones so easiest ones to check incone are first
mutable struct Cone
    prms::Vector{PrimitiveCone}
    idxs::Vector{AbstractVector{Int}}
end
Cone() = Cone(PrimitiveCone[], AbstractVector{Int}[])

function addprimitivecone!(cone::Cone, prm::PrimitiveCone, idx::AbstractVector{Int})
    @assert dimension(prm) == length(idx)
    push!(cone.prms, prm)
    push!(cone.idxs, idx)
    return cone
end

# calculate complexity parameter of the barrier (sum of the primitive cone barrier parameters)
barrierpar(cone::Cone) = sum(barrierpar_prm(prm) for prm in cone.prms)

function getintdir!(dir::Vector{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        getintdir_prm!(view(dir, cone.idxs[k]), cone.prms[k])
    end
    return dir
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

function calcHarr!(prod::AbstractMatrix{Float64}, arr::AbstractMatrix{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcHarr_prm!(view(prod, cone.idxs[k], :), view(arr, cone.idxs[k], :), cone.prms[k])
    end
    return prod
end

function calcHarr!(prod::AbstractVector{Float64}, arr::AbstractVector{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcHarr_prm!(view(prod, cone.idxs[k]), view(arr, cone.idxs[k]), cone.prms[k])
    end
    return prod
end

function calcHiarr!(prod::AbstractMatrix{Float64}, arr::AbstractMatrix{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcHiarr_prm!(view(prod, cone.idxs[k], :), view(arr, cone.idxs[k], :), cone.prms[k])
    end
    return prod
end

function calcHiarr!(prod::AbstractVector{Float64}, arr::AbstractVector{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcHiarr_prm!(view(prod, cone.idxs[k]), view(arr, cone.idxs[k]), cone.prms[k])
    end
    return prod
end

# utilities for converting between smat and svec forms (lower triangle) for symmetric matrices
# TODO only need to do lower triangle if use symmetric matrix types
const rt2 = sqrt(2)
const rt2i = inv(rt2)

function mattovec!(vec::AbstractVector{Float64}, mat::AbstractMatrix{Float64})
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
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
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            mat[i,j] = vec[k]
        else
            mat[i,j] = mat[j,i] = rt2i*vec[k]
        end
        k += 1
    end
    return mat
end
