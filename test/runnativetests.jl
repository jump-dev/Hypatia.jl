#=
run native instance tests from test/nativeinstances.jl and display basic benchmarks
=#

using Test
using Printf
using DataFrames
import Hypatia
import Hypatia.Solvers
include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "nativesets.jl"))

# common solver options
# tol = 1e-10
common_options = (
    # verbose = true,
    verbose = false,
    iter_limit = 100,
    time_limit = 6e1,
    # tol_feas = tol,
    # tol_rel_opt = tol,
    # tol_abs_opt = tol,
    )

all_reals = [
    Float64,
    # Float32,
    # BigFloat,
    ]
default_reals = [
    Float64,
    ]

string_nameof(T) = string(nameof(T))
type_name(::T) where {T} = string_nameof(T)

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
        push!(perf, (inst_name, string(T), type_name(solver.stepper), type_name(solver.system_solver), solver.init_use_indirect, solver.preprocess, solver.reduce, test_time, string(Solvers.get_status(solver))))
        @printf("%8.2e seconds\n", test_time)
    end
    return nothing
end

perf = DataFrame(
    inst_name = String[],
    real_T = String[],
    stepper = String[],
    system_solver = String[],
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
        test_instance_solver(inst_name, Float64, common_options)
    end
end

@testset "steppers tests" begin
    println("\nstarting steppers tests")
    steppers = [
        # (Solvers.HeurCombStepper, all_reals), # default
        (Solvers.PredOrCorrStepper, all_reals),
        ]
    for inst_name in inst_cones_few, (stepper, real_types) in steppers, T in real_types
        options = (; common_options..., stepper = stepper{T}())
        test_instance_solver(inst_name, T, options, string_nameof(stepper))
    end
end

@testset "system solvers tests" begin
    println("\nstarting system solvers tests")
    system_solvers = [
        (Solvers.NaiveDenseSystemSolver, all_reals),
        (Solvers.NaiveSparseSystemSolver, default_reals),
        (Solvers.NaiveElimDenseSystemSolver, all_reals),
        (Solvers.NaiveElimSparseSystemSolver, default_reals),
        (Solvers.SymIndefDenseSystemSolver, all_reals),
        (Solvers.SymIndefSparseSystemSolver, default_reals),
        (Solvers.QRCholDenseSystemSolver, all_reals),
        ]
    for inst_name in inst_cones_few, (system_solver, real_types) in system_solvers, T in real_types
        options = (; common_options..., system_solver = system_solver{T}(), reduce = false)
        test_instance_solver(inst_name, T, options, string_nameof(system_solver))
    end
end

@testset "indirect solvers tests" begin
    println("\nstarting indirect solvers tests")
    for inst_name in inst_indirect, T in all_reals
        options = (; common_options..., init_use_indirect = true, preprocess = false, reduce = false, system_solver = Solvers.SymIndefIndirectSystemSolver{T}(), tol_feas = 1e-3, tol_rel_opt = 1e-3, tol_abs_opt = 1e-3)
        test_instance_solver(inst_name, T, options)
    end
end

@testset "no preprocess tests" begin
    println("\nstarting no preprocess tests")
    for inst_name in inst_cones_few, T in all_reals
        options = (; common_options..., preprocess = false, reduce = false, system_solver = Solvers.SymIndefDenseSystemSolver{T}())
        test_instance_solver(inst_name, T, options)
    end
end

# show(perf, allrows = true, allcols = true)
# println("\n")
end
;
