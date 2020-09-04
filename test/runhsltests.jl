#=
run tests using HSL sparse linear system solver caches
requires that HSL.jl be installed and built successfully; Requires.jl handles this optional dependency
=#

import HSL
import Hypatia
import Hypatia.Solvers

include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "common_nativetests.jl"))

hsl_sys_name = "SymIndefSparse"

blas_reals = [
    Float64,
    Float32,
    ]

@info("starting HSL cache tests")
@testset "HSL cache tests" begin
    @testset "cache setup tests: $cache_type" for cache_type in [Hypatia.HSLSymCache] # TODO wrap and test a HSLNonSymCache
        cache = cache_type()
        @test isa(cache, cache_type{Float64})
        @test cache.analyzed == false
        @test cache_type{Float32}().analyzed == false
        @test_throws Exception cache_type{BigFloat}() # TODO make error more specific
    end

    @testset "SymIndefSparse tests: $inst_name, $T" for inst_name in inst_cones_many, T in blas_reals
        run_instance_options(T, inst_name, hsl_sys_name, "$inst_name system_solver = SymIndefSparse $T", system_solver_options = (fact_cache = Hypatia.HSLSymCache{T}(),))
    end
end
