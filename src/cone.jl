
#=
cone object
=#

# TODO reorder primitive cones so easiest ones to check incone are first
mutable struct Cone
    prms::Vector{PrimitiveCone}
    idxs::Vector{AbstractUnitRange}
end

# calculate complexity parameter of the barrier (sum of the primitive cone barrier parameters)
barrierpar(cone::Cone) = sum(barrierpar_prm(prm) for prm in cone.prms)

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

function calcHiprod!(Hi_mat::AbstractMatrix{Float64}, mat::AbstractMatrix{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcHiprod_prm!(view(Hi_mat, cone.idxs[k], :), view(mat, cone.idxs[k], :), cone.prms[k])
    end
    return Hi_mat
end

function calcHiprod!(Hi_vec::Vector{Float64}, vec::Vector{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcHiprod_prm!(view(Hi_vec, cone.idxs[k]), view(vec, cone.idxs[k]), cone.prms[k])
    end
    return Hi_vec
end

function calcLiprod!(Li_mat::AbstractMatrix{Float64}, mat::AbstractMatrix{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcLiprod_prm!(view(Li_mat, cone.idxs[k], :), view(mat, cone.idxs[k], :), cone.prms[k])
    end
    return Li_mat
end

function calcLiprod!(Li_vec::Vector{Float64}, vec::Vector{Float64}, cone::Cone)
    for k in eachindex(cone.prms)
        calcLiprod_prm!(view(Li_vec, cone.idxs[k]), view(vec, cone.idxs[k]), cone.prms[k])
    end
    return Li_vec
end
