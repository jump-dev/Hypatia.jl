


# function predict_then_correct(point::Models.Point, residual::Models.Point, mu::Float64, solver::HSDSolver)
#     cones = solver.model.cones
#
#     # prediction phase
#     direction = get_prediction_direction(point, residual, solver)
#     alpha = 0.9999 * get_max_alpha(point, direction, solver)
#     point = step_in_direction(point, direction, alpha)
#
#     # correction phase
#     num_corr_steps = 0
#     while nbhd > eta && num_corr_steps < solver.max_corr_steps
#         direction = get_correction_direction(point, solver)
#         point = step_in_direction(point, direction, 0.9999)
#         for k in eachindex(cones)
#             @assert Cones.check_in_cone(cones[k], point.primal_views[k])
#         end
#         num_corr_steps += 1
#     end
#     if num_corr_steps == solver.max_corr_steps
#         solver.verbose && println("corrector phase finished outside the eta-neighborhood; terminating")
#         solver.status = :CorrectorFail
#         return (false, point)
#     end
#
#     return (true, point)
# end
