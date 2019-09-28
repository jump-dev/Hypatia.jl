#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

import HSL

mutable struct HSLSymCache{T <: Real} <: SparseSymCache{T}
    analyzed::Bool
    ma57::Pardiso.PardisoSolver
    function PardisoNonSymCache{Float64}()
        cache = new{Float64}()
        cache.analyzed = false
        return cache
    end
end
HSLSymCache{T}() where {T <: Real} = error("Pardiso only works with real type Float64 or Float32")
HSLSymCache() = HSLSymCache{Union{Float64, Float32}}()

function update_sparse_fact(cache::HSLSymCache, A::SparseMatrixCSC{<:HSL.Ma57Data, Int})
    if !cache.analyzed
        cache.ma57 = HSL.Ma57(A)
        cache.analyzed = true
    end
    HSL.ma57_factorize(cache.ma57)
    return
end

function solve_sparse_system(cache::HSLSymCache, x::Matrix{Float64}, A::SparseMatrixCSC{<:HSL.Ma57Data, Int}, b::Matrix{Float64})
    ma57 = cache.ma57
    x .= ma57_solve(ma57, b)
    return x
end

free_memory(cache::HSLSymCache) = nothing
