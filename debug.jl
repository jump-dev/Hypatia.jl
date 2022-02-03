using JuMP, Hypatia

model = Model(Hypatia.Optimizer)
# @variable(model, x)
@variable(model, x)
optimize!(model)

#
# backend = JuMP.backend(model)
# MOI.Utilities.attach_optimizer(backend)
#
# # something not right here-
# MOI.copy_to(backend.optimizer, backend.model_cache, copy_names = false)
# MOIU.automatic_copy_to(backend.optimizer, backend.model_cache; copy_names = false)
# b = backend.optimizer
# c = b.model
# MOI.optimize!(c.optimizer) # fine
# MOI.optimize!(backend) # messes up
