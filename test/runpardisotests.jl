#=
run tests using Pardiso sparse linear system solver caches
requires that Pardiso.jl be installed and built successfully; Requires.jl handles this optional dependency
=#

import Pardiso
import Hypatia.Solvers

include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "nativesets.jl"))

options = (verbose = true,)

@info("starting Pardiso cache tests")
@testset "Pardiso cache tests" begin
    @testset "cache setup tests: $cache_type" for cache_type in [Hypatia.PardisoSymCache, Hypatia.PardisoNonSymCache]
        cache = cache_type()
        @test isa(cache, cache_type{Float64})
        @test cache.analyzed == false
        @test_throws Exception cache_type{Float32}() # TODO make error more specific
    end

    T = Float64
    @testset "NaiveSparse tests: $inst_name" for inst_name in inst_cones_many
        inst_function = eval(Symbol(inst_name))
        inst_function(T, solver = Solvers.Solver{T}(system_solver = Solvers.NaiveSparseSystemSolver{T}(fact_cache = Hypatia.PardisoNonSymCache{T}()); options...))
    end
    @testset "NaiveElimSparse tests: $inst_name" for inst_name in inst_cones_many
        inst_function = eval(Symbol(inst_name))
        inst_function(T, solver = Solvers.Solver{T}(system_solver = Solvers.NaiveElimSparseSystemSolver{T}(fact_cache = Hypatia.PardisoNonSymCache{T}()); options...))
    end
    @testset "SymIndefSparse tests: $inst_name" for inst_name in inst_cones_many
        inst_function = eval(Symbol(inst_name))
        inst_function(T, solver = Solvers.Solver{T}(system_solver = Solvers.SymIndefSparseSystemSolver{T}(fact_cache = Hypatia.PardisoSymCache{T}()); options...))
    end
end
