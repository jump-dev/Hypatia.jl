#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "native.jl"))

const SO = Hypatia.Solvers

real_types = [
    Float64,
    # Float32,
    # BigFloat,
    ]

system_solvers = [
    # SO.QRCholDenseSystemSolver,
    # SO.SymIndefDenseSystemSolver,
    # SO.SymIndefSparseSystemSolver,
    # SO.NaiveElimDenseSystemSolver,
    # SO.NaiveElimSparseSystemSolver,
    # SO.NaiveDenseSystemSolver,
    SO.NaiveSparseSystemSolver,
    # SO.NaiveIndirectSystemSolver,
    ]

cache_dict = Dict(
    SO.QRCholDenseSystemSolver => [nothing],
    SO.SymIndefDenseSystemSolver => [nothing],
    SO.SymIndefSparseSystemSolver => [
        Hypatia.CHOLMODSymCache{Float64}(diag_pert = sqrt(eps())),
        Hypatia.PardisoSymCache(diag_pert = 0.0),
        Hypatia.PardisoSymCache(diag_pert = sqrt(eps())),
        ],
    SO.NaiveDenseSystemSolver => [nothing],
    SO.NaiveSparseSystemSolver => [
        Hypatia.UMFPACKNonSymCache(),
        # Hypatia.PardisoNonSymCache(),
        ],
    SO.NaiveElimDenseSystemSolver => [nothing],
    SO.NaiveElimSparseSystemSolver => [
        Hypatia.UMFPACKNonSymCache(),
        # Hypatia.PardisoNonSymCache(),
        ],
    SO.NaiveIndirectSystemSolver => [nothing],
    )

options_dict = Dict(
    SO.QRCholDenseSystemSolver => [NamedTuple()],
    SO.SymIndefDenseSystemSolver => [NamedTuple()],
    SO.SymIndefSparseSystemSolver => [
        (use_inv_hess = true,)
        ],
    SO.NaiveDenseSystemSolver => [NamedTuple()],
    SO.NaiveSparseSystemSolver => [NamedTuple()],
    SO.NaiveElimDenseSystemSolver => [
        (use_inv_hess = true,),
        (use_inv_hess = false,),
        ],
    SO.NaiveElimSparseSystemSolver => [
        (use_inv_hess = true,),
        ],
    SO.NaiveIndirectSystemSolver => [NamedTuple()],
    )

use_infty_nbhd = [
    true,
    # false,
    ]

preprocess = [
    true,
    # false
    ]

testfuns_preproc = [
    dimension1,
    consistent1,
    inconsistent1,
    inconsistent2,
    ]

testfuns_raw = [
    orthant1,
    orthant2,
    orthant3,
    orthant4,
    epinorminf1,
    epinorminf2,
    epinorminf3,
    epinorminf4,
    epinorminf5,
    epinormeucl1,
    epinormeucl2,
    epipersquare1,
    epipersquare2,
    epipersquare3,
    hypoperlog1,
    hypoperlog2,
    hypoperlog3,
    hypoperlog4,
    hypoperlog5,
    hypoperlog6,
    epiperexp1,
    epiperexp2,
    power1,
    power2,
    power3,
    power4,
    hypogeomean1,
    hypogeomean2,
    hypogeomean3,
    epinormspectral1,
    possemideftri1,
    possemideftri2,
    possemideftricomplex1,
    hypoperlogdettri1,
    hypoperlogdettri2,
    hypoperlogdettri3,
    primalinfeas1,
    primalinfeas2,
    primalinfeas3,
    dualinfeas1,
    dualinfeas2,
    dualinfeas3,
    ]

tol = 1e-8 # TODO delete later

@info("starting native tests")
@testset "native tests" begin
    # @info("starting preprocessing tests")
    # @testset "preprocessing tests: $t, $T" for t in testfuns_preproc, T in real_types
    #     t(T, solver = SO.Solver{T}(verbose = true, system_solver = SO.QRCholDenseSystemSolver{T}()))
    # end

    @info("starting miscellaneous tests")
    @testset "miscellaneous tests: $t, $s, $n, $p, $T" for t in testfuns_raw, s in system_solvers, n in use_infty_nbhd, p in preprocess, T in real_types
        !p && s == SO.QRCholSystemSolver && continue # must use preprocessing if using QRCholSystemSolver
        T == BigFloat && t == epinormspectral1 && continue # cannot get svdvals with BigFloat
        T == BigFloat && s == SO.NaiveIndirectSystemSolver && continue # cannot use indirect methods with BigFloat
        T != Float64 && s in (SO.SymIndefSparseSystemSolver, SO.NaiveSparseSystemSolver) && continue # sparse system solvers only work with Float64
        caches = cache_dict[s]
        options = options_dict[s]
        @testset "$ci, $oi" for (ci, c) in enumerate(caches), (oi, o) in enumerate(options)
            system_solver = (c === nothing ? system_solver = s{T}(; o...) : system_solver = s{T}(; fact_cache = c, o...))
            solver = SO.Solver{T}(verbose = true, preprocess = p, use_infty_nbhd = n,
                system_solver = system_solver, tol_abs_opt = tol, tol_rel_opt = tol, tol_feas = tol)
            t(T, solver = solver)
        end
    end
end
