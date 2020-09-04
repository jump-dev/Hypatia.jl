#=
sets of native test instances and helper functions
=#

# other solver options
timer = TimerOutput()
# tol = 1e-10
# tol = 1e-7
other_options = (
    verbose = true,
    # verbose = false,
    iter_limit = 100,
    time_limit = 6e1,
    timer = timer,
    # tol_feas = tol,
    # tol_rel_opt = tol,
    # tol_abs_opt = tol,
    # rescale = false,
    # preprocess = false,
    # reduce = false,
    )

perf = DataFrame(
    inst_name = String[],
    sys_solver = String[],
    real_T = String[],
    preprocess = Bool[],
    init_use_indirect = Bool[],
    reduce = Bool[],
    test_time = Float64[],
    )

function run_instance_options(T::Type{<:Real}, inst_name::String, sys_name::String, test_info::String; system_solver_options = NamedTuple(), kwargs...)
    @testset "$test_info" begin
        println(test_info, "...")
        inst_function = eval(Symbol(inst_name))
        sys_solver = Solvers.eval(Symbol(sys_name, "SystemSolver"))
        solver = Solvers.Solver{T}(; system_solver = sys_solver{T}(; system_solver_options...), kwargs..., other_options...)
        test_time = @elapsed inst_function(T, solver = solver)
        push!(perf, (inst_name, sys_name, string(T), solver.preprocess, solver.init_use_indirect, solver.reduce, test_time))
        @printf("... %8.2e seconds\n", test_time)
    end
    return nothing
end

inst_preproc = [ # TODO add more preprocessing test instances
    "dimension1",
    "consistent1",
    "inconsistent1",
    "inconsistent2",
    ]

inst_infeas = [
    "primalinfeas1",
    "primalinfeas2",
    "primalinfeas3",
    "dualinfeas1",
    "dualinfeas2",
    "dualinfeas3",
    ]

inst_cones_few = [
    "nonnegative1",
    "epinorminf1",
    "epinorminf6",
    "epinormeucl1",
    "epipersquare1",
    "episumperentropy1",
    "hypoperlog1",
    "power1",
    "hypogeomean1",
    "hypopowermean1",
    "epinormspectral1",
    "matrixepipersquare1",
    "linmatrixineq1",
    "possemideftri1",
    "possemideftri5",
    "possemideftrisparse2",
    "possemideftrisparse5",
    "doublynonnegative1",
    "hypoperlogdettri1",
    "hyporootdettri1",
    "wsosinterpnonnegative1",
    "wsosinterppossemideftri1",
    "wsosinterpepinormeucl1",
    ]

inst_cones_many = [
    "nonnegative1",
    "nonnegative2",
    "nonnegative3",
    "nonnegative4",
    "epinorminf1",
    "epinorminf2",
    "epinorminf3",
    "epinorminf4",
    "epinorminf5",
    "epinorminf6",
    "epinorminf7",
    "epinorminf8",
    "epinormeucl1",
    "epinormeucl2",
    "epinormeucl3",
    "epipersquare1",
    "epipersquare2",
    "epipersquare3",
    "epipersquare4",
    "episumperentropy1",
    "episumperentropy2",
    "episumperentropy3",
    "episumperentropy4",
    "episumperentropy5",
    "episumperentropy6",
    "hypoperlog1",
    "hypoperlog2",
    "hypoperlog3",
    "hypoperlog4",
    "hypoperlog5",
    "hypoperlog6",
    "hypoperlog7",
    "hypogeomean1",
    "hypogeomean2",
    "hypogeomean3",
    "hypogeomean4",
    "hypogeomean5",
    "hypogeomean6",
    "hypopowermean1",
    "hypopowermean2",
    "hypopowermean3",
    "hypopowermean4",
    "hypopowermean5",
    "hypopowermean6",
    "power1",
    "power2",
    "power3",
    "power4",
    "epinormspectral1",
    "epinormspectral2",
    "epinormspectral3",
    "epinormspectral4",
    "linmatrixineq1",
    "linmatrixineq2",
    "linmatrixineq3",
    "possemideftri1",
    "possemideftri2",
    "possemideftri3",
    "possemideftri4",
    "possemideftri5",
    "possemideftri6",
    "possemideftri7",
    "possemideftri8",
    "possemideftri9",
    "possemideftrisparse1",
    "possemideftrisparse2",
    "possemideftrisparse3",
    "possemideftrisparse4",
    "possemideftrisparse5",
    "doublynonnegative1",
    "doublynonnegative2",
    "doublynonnegative3",
    "matrixepipersquare1",
    "matrixepipersquare2",
    "matrixepipersquare3",
    "hypoperlogdettri1",
    "hypoperlogdettri2",
    "hypoperlogdettri3",
    "hypoperlogdettri4",
    "hyporootdettri1",
    "hyporootdettri2",
    "hyporootdettri3",
    "hyporootdettri4",
    "wsosinterpnonnegative1",
    "wsosinterpnonnegative2",
    "wsosinterpnonnegative3",
    "wsosinterppossemideftri1",
    "wsosinterppossemideftri2",
    "wsosinterppossemideftri3",
    "wsosinterpepinormeucl1",
    "wsosinterpepinormeucl2",
    "wsosinterpepinormeucl3",
    ]
