#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "barrier.jl"))

@info("starting barrier tests")

real_types = [
    Float64,
    Float32,
    BigFloat, # NOTE can only use BLAS floats with ForwardDiff barriers, see https://github.com/JuliaDiff/DiffResults.jl/pull/9#issuecomment-497853361
    ]

barrier_testfuns = [
    # test_orthant_barrier,
    # test_epinorminf_barrier,
    # test_epinormeucl_barrier,
    # test_epipersquare_barrier,
    # test_epiperpower_barrier, # fails with BigFloat
    # test_hypoperlog_barrier,
    # test_epiperexp_barrier, # fails with BigFloat
    # test_hypogeomean_barrier,
    # test_epinormspectral_barrier,
    test_possemideftri_barrier,
    # test_hypoperlogdettri_barrier,
    # # test_wsospolyinterp_barrier,
    # # TODO next 2 fail with BigFloat
    # # NOTE not updated for generic reals or for new cone oracles interface
    # # test_wsospolyinterpmat_barrier,
    # # test_wsospolyinterpsoc_barrier,
    ]

@testset "barrier functions tests: $t, $T" for t in barrier_testfuns, T in real_types
    if T == BigFloat && t in (test_epiperpower_barrier, test_epiperexp_barrier) #, test_wsospolyinterpmat_barrier, test_wsospolyinterpsoc_barrier)
        continue
    end
    t(T)
end
