#=
run native instance tests from test/nativeinstances.jl and display basic benchmarks
=#

using Test
using DataFrames
using Printf
import Hypatia
import Hypatia.Solvers

include(joinpath(@__DIR__, "nativeinstances.jl"))
include(joinpath(@__DIR__, "nativesets.jl"))

# common solver options
# tol = 1e-10
common_options = (
    # verbose = true,
    verbose = false,
    iter_limit = 20,
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
    test_info::String,
    )
    test_info = "$inst_name $T $test_info"
    println("\n", test_info, " ...")
    @testset "$test_info" begin
        solver = Solvers.Solver{T}(; options...)
        test_time = @elapsed eval(Symbol(inst_name))(T, solver = solver)
        push!(perf, (inst_name, string(T), type_name(solver.stepper), type_name(solver.system_solver), solver.init_use_indirect, solver.preprocess, solver.reduce, test_time, string(Solvers.get_status(solver))))
        @printf("... %8.2e seconds\n", test_time)
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

@info("starting native tests")

all_tests_time = time()
global ITERS = 0

@testset "native tests" begin

@testset "default options tests" begin
for inst_name in vcat(inst_preproc, inst_infeas, inst_cones_many)
    test_instance_solver(inst_name, Float64, common_options, "defaults")
end
end

# @testset "steppers tests" begin
# steppers = [
#     # (Solvers.HeurCombStepper, all_reals), # default
#     (Solvers.PredOrCorrStepper, all_reals),
#     ]
# for inst_name in inst_cones_few, (stepper, real_types) in steppers, T in real_types
#     test_info = "stepper = $(string_nameof(stepper))"
#     options = (; common_options..., stepper = stepper{T}())
#     test_instance_solver(inst_name, T, options, test_info)
# end
# end
#
# @testset "system solvers tests" begin
# system_solvers = [
#     (Solvers.NaiveDenseSystemSolver, all_reals),
#     (Solvers.NaiveSparseSystemSolver, default_reals),
#     (Solvers.NaiveElimDenseSystemSolver, all_reals),
#     (Solvers.NaiveElimSparseSystemSolver, default_reals),
#     (Solvers.SymIndefDenseSystemSolver, all_reals),
#     (Solvers.SymIndefSparseSystemSolver, default_reals),
#     (Solvers.QRCholDenseSystemSolver, all_reals),
#     ]
# for inst_name in inst_cones_few, (system_solver, real_types) in system_solvers, T in real_types
#     test_info = "system_solver = $(string_nameof(system_solver))"
#     options = (; common_options..., system_solver = system_solver{T}(), reduce = false)
#     test_instance_solver(inst_name, T, options, test_info)
# end
# end
#
# @testset "indirect solvers tests" begin
# for inst_name in inst_indirect, T in all_reals
#     test_info = "indirect solvers"
#     options = (; common_options..., init_use_indirect = true, preprocess = false, reduce = false, system_solver = Solvers.SymIndefIndirectSystemSolver{T}(), tol_feas = 1e-3, tol_rel_opt = 1e-3, tol_abs_opt = 1e-3)
#     test_instance_solver(inst_name, T, options, test_info)
# end
# end
#
# @testset "no preprocess tests" begin
# for inst_name in inst_cones_few, T in all_reals
#     test_info = "preprocess = false"
#     options = (; common_options..., preprocess = false, reduce = false, system_solver = Solvers.SymIndefDenseSystemSolver{T}())
#     test_instance_solver(inst_name, T, options, test_info)
# end
# end
#
# @printf("\nnative tests total time: %8.2e seconds\n\n", time() - all_tests_time)
# show(perf, allrows = true, allcols = true)
# println("\n")
end
;
