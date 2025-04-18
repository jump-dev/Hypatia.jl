#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
utilities for Pardiso
only works with Float64

TODO add to docs following example:
```julia
import Hypatia

ENV["OMP_NUM_THREADS"] = length(Sys.cpu_info())
import Pardiso

syssolver = Hypatia.Solvers.NaiveElimSparseSystemSolver{Float64}(fact_cache = Hypatia.PardisoNonSymCache())
solver = Hypatia.Solvers.Solver{Float64}(syssolver)

include("test/native.jl")
nonnegative1(Float64, solver = solver)
```
=#

mutable struct PardisoNonSymCache <: SparseNonSymCache{Float64}
    analyzed::Bool
    pardiso::Pardiso.MKLPardisoSolver
    function PardisoNonSymCache()
        cache = new()
        cache.analyzed = false
        cache.pardiso = Pardiso.MKLPardisoSolver()
        Pardiso.set_matrixtype!(cache.pardiso, Pardiso.REAL_NONSYM)
        return cache
    end
end

mutable struct PardisoSymCache <: SparseSymCache{Float64}
    analyzed::Bool
    pardiso::Pardiso.MKLPardisoSolver
    function PardisoSymCache()
        cache = new()
        cache.analyzed = false
        cache.pardiso = Pardiso.MKLPardisoSolver()
        Pardiso.set_matrixtype!(cache.pardiso, Pardiso.REAL_SYM_INDEF)
        return cache
    end
end

const PardisoSparseCache = Union{PardisoSymCache, PardisoNonSymCache}

int_type(::PardisoSparseCache) = Int32

function update_fact(cache::PardisoSparseCache, A::SparseMatrixCSC{Float64, Int32})
    pardiso = cache.pardiso

    if !cache.analyzed
        Pardiso.pardisoinit(pardiso)
        # solve transposed problem (Pardiso accepts CSR matrices)
        Pardiso.fix_iparm!(pardiso, :N)
        Pardiso.set_phase!(pardiso, Pardiso.ANALYSIS)
        Pardiso.pardiso(pardiso, A, Float64[])
        cache.analyzed = true
    end

    Pardiso.set_phase!(pardiso, Pardiso.NUM_FACT)
    Pardiso.pardiso(pardiso, A, Float64[])

    return
end

function inv_prod(
    cache::PardisoSparseCache,
    x::Vector{Float64},
    A::SparseMatrixCSC{Float64, Int32},
    b::Vector{Float64},
)
    pardiso = cache.pardiso

    Pardiso.set_phase!(pardiso, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(pardiso, x, A, b)

    return x
end

function free_memory(cache::PardisoSparseCache)
    Pardiso.set_phase!(cache.pardiso, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(cache.pardiso)
    return
end
