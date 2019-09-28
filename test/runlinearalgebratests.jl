#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

@testset "PardisoCache" begin
    cache = Hypatia.PardisoSymCache()
    @test cache.analyzed == false
    @test isa(cache, Hypatia.PardisoSymCache{Float64})
    @test_throws Exception Hypatia.PardisoSymCache{Float32}()

    cache = Hypatia.PardisoNonSymCache()
    @test cache.analyzed == false
    @test isa(cache, Hypatia.PardisoNonSymCache{Float64})
    @test_throws Exception Hypatia.PardisoNonSymCache{Float32}()
end
