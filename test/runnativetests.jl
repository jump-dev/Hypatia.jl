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

perf = DataFrames.DataFrame(
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
        test_instance_solver(inst_name, Float64, default_options)
    end
end

@testset "no preprocess tests" begin
    println("\nstarting no preprocess tests")
    for inst_name in inst_cones_few, T in diff_reals
        options = (; default_options..., preprocess = false, reduce = false, system_solver = Solvers.SymIndefDenseSystemSolver{T}())
        test_instance_solver(inst_name, T, options)
    end
end

@testset "indirect solvers tests" begin
    println("\nstarting indirect solvers tests")
    for inst_name in inst_indirect, T in diff_reals
        options = (; default_options..., init_use_indirect = true, preprocess = false, reduce = false, system_solver = Solvers.SymIndefIndirectSystemSolver{T}(), tol_feas = 1e-4, tol_rel_opt = 1e-4, tol_abs_opt = 1e-4)
        test_instance_solver(inst_name, T, options)
    end
end

@testset "system solvers tests" begin
    println("\nstarting system solvers tests")
    system_solvers = [
        (Solvers.NaiveDenseSystemSolver, diff_reals),
        (Solvers.NaiveSparseSystemSolver, [Float64,]),
        (Solvers.NaiveElimDenseSystemSolver, diff_reals),
        (Solvers.NaiveElimSparseSystemSolver, [Float64,]),
        (Solvers.SymIndefDenseSystemSolver, all_reals),
        (Solvers.SymIndefSparseSystemSolver, [Float64,]),
        (Solvers.QRCholDenseSystemSolver, all_reals),
        ]
    for inst_name in inst_cones_few, (system_solver, real_types) in system_solvers, T in real_types
        options = (; default_options..., system_solver = system_solver{T}(), reduce = false)
        test_instance_solver(inst_name, T, options, string_nameof(system_solver))
    end
end

@testset "steppers tests" begin
    println("\nstarting steppers tests")
    steppers = [
        (Solvers.HeurCombStepper, diff_reals),
        (Solvers.PredOrCentStepper, diff_reals),
        ]
    for inst_name in inst_cones_few, (stepper, real_types) in steppers, T in diff_reals
        options = (; default_options..., stepper = stepper{T}())
        test_instance_solver(inst_name, T, options, string_nameof(stepper))
    end
end

# println("\n")
# DataFrames.show(perf, allrows = true, allcols = true)
# println("\n")
end
;
