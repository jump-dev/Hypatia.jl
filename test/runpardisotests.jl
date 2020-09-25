#=
run tests using Pardiso sparse linear system solver caches
requires that PardiSolvers.jl be installed and built successfully; Requires.jl handles this optional dependency
=#

using Test
import Pardiso
import Hypatia
import Hypatia.Solvers
include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "nativesets.jl"))

options = (verbose = false,)

@testset "Pardiso cache tests" begin

@testset "cache setup: $cache_type" for cache_type in [Hypatia.PardisoSymCache, Hypatia.PardisoNonSymCache]
    cache = cache_type()
    @test !cache.analyzed
    @test_throws Exception cache_type{Float32}()
end

T = Float64
@testset "NaiveSparse: $t" for t in testfuns_many
    t(T, solver = Solvers.Solver{T}(system_solver = Solvers.NaiveSparseSystemSolver{T}(fact_cache = Hypatia.PardisoNonSymCache()); options...))
end
@testset "NaiveElimSparse: $t" for t in testfuns_many
    t(T, solver = Solvers.Solver{T}(system_solver = Solvers.NaiveElimSparseSystemSolver{T}(fact_cache = Hypatia.PardisoNonSymCache()); options...))
end
@testset "SymIndefSparse: $t" for t in testfuns_many
    t(T, solver = Solvers.Solver{T}(system_solver = Solvers.SymIndefSparseSystemSolver{T}(fact_cache = Hypatia.PardisoSymCache()); options...))
end

end
