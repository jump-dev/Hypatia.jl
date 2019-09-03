#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "moi.jl"))

verbose = false

real_types = [
    Float64,
    # BigFloat, # TODO test this when MOI allows
    ]

dense_options = [
    true,
    # false,
    ]

system_solvers = [
    SO.QRCholCombinedHSDSystemSolver,
    # SO.SymIndefCombinedHSDSystemSolver,
    # SO.NaiveElimCombinedHSDSystemSolver,
    # SO.NaiveCombinedHSDSystemSolver,
    ]

@info("starting MOI tests")
@testset "MOI tests: $(d ? "dense" : "sparse"), $s, $T" for d in dense_options, s in system_solvers, T in real_types
    hypatia_options = (
        # linear_model = m,
        system_solver = s,
        # linear_model_options = NamedTuple(),
        # system_solver_options = NamedTuple(),
        stepper_options = NamedTuple(),
        solver_options = (verbose = false,),
        )

    # test_moi(d, s{T}, MO.PreprocessedLinearModel{T}, verbose)
    test_moi(T, d, hypatia_options)
end
