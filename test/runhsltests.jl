#=
run tests using HSL sparse linear system solver caches
requires that HSL.jl be installed and built successfully; Requires.jl handles this optional dependency
=#

using Test
import HSL
import Hypatia
import Hypatia.Solvers
blas_reals = [
    Float64,
    Float32,
    ]
include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "nativesets.jl"))

options = (verbose = false,)

@testset "HSL cache tests" begin

@testset "cache setup: $cache_type" for cache_type in [Hypatia.HSLSymCache] # TODO wrap and test a HSLNonSymCache
    cache = cache_type()
    @test !cache.analyzed
    @test cache_type{Float32}().analyzed == false
    @test_throws Exception cache_type{BigFloat}()
end

@testset "SymIndefSparse tests: $inst_name, $T" for inst_name in inst_cones_many,
    T in blas_reals
    inst_function = eval(Symbol(inst_name))
    inst_function(T, solver = Solvers.Solver{T}(syssolver =
        Solvers.SymIndefSparseSystemSolver{T}(
        fact_cache = Hypatia.HSLSymCache{T}()); options...))
end

end
;
