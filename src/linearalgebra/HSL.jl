#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

mutable struct HSLSymCache{T <: Union{Float32, Float64}} <: SparseSymCache{T}
    analyzed::Bool
    ma57::HSL.Ma57{T}
    diag_pert
    function HSLSymCache{T}(; diag_pert = sqrt(eps(T))) where {T <: Union{Float32, Float64}}
        cache = new{T}()
        cache.analyzed = false
        cache.diag_pert = diag_pert
        return cache
    end
end
HSLSymCache{T}(; diag_pert = zero(T)) where {T <: Real} = error("HSL only works with real type Float64 or Float32")
HSLSymCache(; diag_pert::Float64 = sqrt(eps(Float64))) = HSLSymCache{Float64}(diag_pert = diag_pert)

int_type(::HSLSymCache) = Int

function update_sparse_fact(cache::HSLSymCache, A::SparseMatrixCSC{<:HSL.Ma57Data, Int})
    if !cache.analyzed
        cache.ma57 = HSL.Ma57(A)
        cache.analyzed = true
    end
    HSL.ma57_factorize(cache.ma57)
    return
end

function solve_sparse_system(cache::HSLSymCache, x::Matrix{Float64}, A::SparseMatrixCSC{<:HSL.Ma57Data, Int}, b::Matrix{Float64})
    # TODO iterative refinement
    ma57 = cache.ma57
    # solve with 2 iterative refinement steps
    HSL.ma57_solve!(ma57, b)
    copyto!(x, b)

    # info we could print
    # ma57.info.backward_error1
    # ma57.info.backward_error2
    # ma57.info.matrix_inf_norm
    # ma57.info.solution_inf_norm
    # ma57.info.scaled_residuals
    # ma57.info.cond1
    # ma57.info.cond2
    # ma57.info.error_inf_norm

    return x
end

free_memory(cache::HSLSymCache) = nothing
