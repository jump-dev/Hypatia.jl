#=
run examples tests from the examples folder
=#

examples_dir = joinpath(@__DIR__, "../examples")
(model_types, native_example_names, JuMP_example_names, perf, test_instances) = include(joinpath(examples_dir, "testaux.jl"))

steppers = [Solvers.CombinedStepper{Float64}()]

@testset "examples tests" begin
@testset "$mod_type" for mod_type in model_types
    @testset "$ex_name" for ex_name in eval(Symbol(mod_type, "_example_names"))
        include(joinpath(examples_dir, ex_name, mod_type * ".jl"))
        (ex_type, ex_insts) = include(joinpath(examples_dir, ex_name, mod_type * "_test.jl"))
        test_instances(steppers, instance_sets, ex_insts, ex_type, perf)
    end
end
# println("\n")
# DataFrames.show(perf, allrows = true, allcols = true)
# println("\n")
# @show sum(perf[!, :iters])
# @show sum(perf[!, :solve_time])
end
