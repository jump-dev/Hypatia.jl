#=
run native instance tests from test/nativeinstances.jl and display basic benchmarks
=#

using Test
using Printf
import DataFrames
import Hypatia
import Hypatia.Solvers
include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "nativesets.jl"))

# default options to solvers
default_options = (
    # verbose = true,
    verbose = false,
    default_tol_relax = 10,
    )

all_reals = [
    Float32,
    Float64,
    BigFloat,
    ]
diff_reals = [
    Float64,
    BigFloat,
    ]

string_nameof(T) = string(nameof(T))

type_name(::T) where T = string_nameof(T)

function test_instance_solver(
    inst_name::String,
    T::Type{<:Real},
    options::NamedTuple,
    test_info::String = "",
    )
    test_info = "$inst_name $T $test_info"
    @testset "$test_info" begin
        println(test_info, " ...")
        solver = Solvers.Solver{T}(; options...)
        test_time = @elapsed eval(Symbol(inst_name))(T, solver = solver)
        push!(perf, (inst_name, string(T), type_name(solver.stepper),
            type_name(solver.syssolver), solver.init_use_indirect,
            solver.preprocess, solver.reduce, test_time,
            string(Solvers.get_status(solver))))
        @printf("%8.2e seconds\n", test_time)
    end
    return nothing
end

perf = DataFrames.DataFrame(
    inst_name = String[],
    real_T = String[],
    stepper = String[],
    syssolver = String[],
    init_use_indirect = Bool[],
    preprocess = Bool[],
    reduce = Bool[],
    test_time = Float64[],
    status = String[],
    )

@testset "native tests" begin

@testset "default options tests" begin
    println("starting default options tests")
    inst_defaults = vcat(
        inst_preproc,
        inst_infeas,
        inst_cones_many,
        )
    for inst_name in inst_defaults
        test_instance_solver(inst_name, Float64, default_options)
    end
end

@testset "no preprocess tests" begin
    println("\nstarting no preprocess tests")
    for inst_name in inst_cones_few, T in diff_reals
        options = (; default_options..., preprocess = false, reduce = false,
            syssolver = Solvers.SymIndefDenseSystemSolver{T}())
        test_instance_solver(inst_name, T, options)
    end
end

@testset "indirect solvers tests" begin
    println("\nstarting indirect solvers tests")
    for inst_name in inst_indirect, T in diff_reals
        options = (; default_options..., init_use_indirect = true,
            preprocess = false, reduce = false,
            syssolver = Solvers.SymIndefIndirectSystemSolver{T}(),
            tol_feas = 1e-4, tol_rel_opt = 1e-4, tol_abs_opt = 1e-4,
            tol_infeas = 1e-6)
        test_instance_solver(inst_name, T, options)
    end
end

@testset "system solvers tests" begin
    println("\nstarting system solvers tests")
    syssolvers = [
        (Solvers.NaiveDenseSystemSolver, diff_reals),
        (Solvers.NaiveSparseSystemSolver, [Float64,]),
        (Solvers.NaiveElimDenseSystemSolver, diff_reals),
        (Solvers.NaiveElimSparseSystemSolver, [Float64,]),
        (Solvers.SymIndefDenseSystemSolver, all_reals),
        (Solvers.SymIndefSparseSystemSolver, [Float64,]),
        (Solvers.QRCholDenseSystemSolver, all_reals),
        ]
    for inst_name in inst_minimal, (syssolver, real_types) in syssolvers,
        T in real_types
        options = (; default_options..., syssolver = syssolver{T}(),
            reduce = false)
        test_instance_solver(inst_name, T, options, string_nameof(syssolver))
    end
end

@testset "PredOrCentStepper tests" begin
    verbose = true
    println("\nstarting PredOrCentStepper tests (with printing)")

    # adjustment and curve search
    use_adj_curv = [(false, false), (true, false), (true, true)]
    for inst_name in inst_minimal, (adj, curv) in use_adj_curv, T in diff_reals
        stepper = Solvers.PredOrCentStepper{T}(;
            use_adjustment = adj, use_curve_search = curv)
        options = (; default_options..., verbose = verbose, stepper = stepper)
        test_instance_solver(inst_name, T, options, "adj=$adj curv=$curv")
    end

    # other options
    for inst_name in inst_minimal
        T = Float64
        stepper = Solvers.PredOrCentStepper{T}(;
            # stepper options
            use_adjustment = false, use_curve_search = false, max_cent_steps = 8,
            pred_prox_bound = 0.0332, use_pred_sum_prox = true,
            # searcher options
            min_prox = 0.0, prox_bound = 0.2844, use_sum_prox = true,
            alpha_sched = [0.9999 * 0.7^i for i in 0:22])
        options = (; default_options..., verbose = verbose, stepper = stepper)
        test_instance_solver(inst_name, T, options, "other")
    end
end

@testset "CombinedStepper tests" begin
    verbose = true
    println("\nstarting CombinedStepper tests (with printing)")
    shifts = [0, 2]
    for inst_name in inst_minimal, shift in shifts, T in diff_reals
        options = (; default_options..., verbose = verbose,
            stepper = Solvers.CombinedStepper{T}(shift_sched = shift))
        test_instance_solver(inst_name, T, options, "shift=$shift")
    end
end

# println("\n")
# DataFrames.show(perf, allrows = true, allcols = true)
# println("\n")
end
;
