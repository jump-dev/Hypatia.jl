#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
utilities for CliqueTrees.Multifrontal
sparse supernodal LDLt factorization
=#

const MF = CliqueTrees.Multifrontal

mutable struct CliqueTreesCache{T <: Real, Reg <: MF.AbstractRegularization} <: SparseSymCache{T}
    analyzed::Bool
    F::MF.FChordalLDLt{:L, T, Int}
    indices::Vector{Int}
    signs::Vector{T}
    reg::Reg

    function CliqueTreesCache{T}(reg::Reg) where {T <: Real, Reg <: MF.AbstractRegularization}
        cache = new{T, Reg}()
        cache.analyzed = false
        cache.reg = reg
        return cache
    end
end

function CliqueTreesCache{T}(; reg::MF.AbstractRegularization=MF.NoRegularization()) where {T <: Real}
    return CliqueTreesCache{T}(reg)
end

int_type(::CliqueTreesCache) = Int

function update_fact(
    cache::CliqueTreesCache{T},
    A::SparseMatrixCSC{T, Int};
    npos,
) where {T <: Real}
    n = size(A, 1)

    if !cache.analyzed
        cache.signs = Vector{T}(undef, n)
        cache.signs[1:npos] .= one(T)
        cache.signs[npos+1:end] .= -one(T)

        cache.F = MF.ChordalLDLt(Symmetric(A, :L))
        cache.indices = MF.flatindices(cache.F, Symmetric(A, :L))
        MF.ldlt!(cache.F; signs=cache.signs, reg=cache.reg, check=false)
        cache.analyzed = true
    else
        # Update values directly via flat indices (avoids full matrix copy)
        nzval = A.nzval
        @inbounds for (i, p) in enumerate(cache.indices)
            if !iszero(p)
                MF.setflatindex!(cache.F, nzval[i], p)
            end
        end
        MF.ldlt!(cache.F; signs=cache.signs, reg=cache.reg, check=false)
    end

    if !issuccess(cache.F)
        @warn("numerical failure: CliqueTrees factorization failed")
    end

    return
end

function inv_prod(
    cache::CliqueTreesCache{T},
    c::Vector{T},
    A::SparseMatrixCSC{T, Int},
    b::Vector{T},
) where {T <: Real}
    return MF.ldiv!(c, cache.F, b)
end

free_memory(::CliqueTreesCache) = nothing
