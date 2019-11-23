#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

see description in maxvolume/native.jl

SOC formulation inspired by MOI bridge
=#

using LinearAlgebra
import JuMP
const MOI = JuMP.MOI
import Hypatia
import Random

function maxvolumeJuMP(
    n::Int;
    constr_cone::Symbol = :soc,
    )

    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@variable(model, end_pts[1:n])
    JuMP.@objective(model, Max, t)
    poly_hrep = Matrix{Float64}(I, n, n)
    poly_hrep .+= randn(n, n) * 10.0 ^ (-n)
    JuMP.@constraint(model, poly_hrep * end_pts .<= ones(n))

    if constr_cone == :geomean
        JuMP.@constraint(model, vcat(t, end_pts) in MOI.GeometricMeanCone(n + 1))
    elseif constr_cone == :soc
        # number of variables inside geometric mean is n
        # number of layers of variables
        # TODO leave as is or copy in function?
        num_layers = MathOptInterface.Bridges.Constraint.ilog2(n)
        # number of new variables = 1 + 2 + ... + 2^(l - 1) = 2^l - 1
        num_new_vars = 2 ^ num_layers - 1
        JuMP.@variable(model, new_vars[1:num_new_vars] >= 0)
        rtfact = sqrt(2 ^ num_layers)
        xl1 = new_vars[1]
        JuMP.@constraint(model, t <= xl1 / rtfact)

        offset = offset_next = 0
        # loop over layers, layer 1 describes hypograph variable
        for i in 1:(num_layers - 1)
            num_lvars = 2 ^ (i - 1)
            offset_next = offset + num_lvars
            # loop over variables in each layer
            for j in 1:num_lvars
                u = new_vars[offset_next + 2j - 1]
                v = new_vars[offset_next + 2j]
                w = new_vars[offset + j]
                JuMP.@constraint(model, [u, v, w] in JuMP.RotatedSecondOrderCone())
            end
            offset = offset_next
        end
        # we are beyond the number of variables new variables, we are in the largest layer
        for j in 1:(2 ^ (num_layers - 1))
            if 2j - 1 > n
                # buffer variable bounds hypograph variable
                u = v = xl1 / rtfact
            else
                # original problem variables
                u = end_pts[2j - 1]
                if 2j > n
                    # buffer variable bounds hypograph variable
                    v = xl1 / rtfact
                else
                    # original problem variables
                    v = end_pts[2j]
                end
            end
            w = new_vars[offset + j]
            JuMP.@constraint(model, [u, v, w] in JuMP.RotatedSecondOrderCone())
        end
    else
        error("unknown cone $(constr_cone)")
    end

    return (model = model,)
end

maxvolumeJuMP1() = maxvolumeJuMP(3, constr_cone = :geomean)
maxvolumeJuMP2() = maxvolumeJuMP(3, constr_cone = :soc)

function test_maxvolumeJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_maxvolumeJuMP_all(; options...) = test_maxvolumeJuMP.([
    maxvolumeJuMP1,
    maxvolumeJuMP2,
    ], options = options)

test_maxvolumeJuMP(; options...) = test_maxvolumeJuMP.([
    maxvolumeJuMP1,
    maxvolumeJuMP2,
    ], options = options)
