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
        l = MathOptInterface.Bridges.Constraint.ilog2(n)
        # number of new variables = 1 + 2 + ... + 2^(l - 1) = 2^l - 1
        N = 2 ^ l
        num_new_vars = N - 1
        JuMP.@variable(model, new_vars[1:num_new_vars] >= 0)
        # JuMP.@constraint(model, new_vars[1] >= 0)
        rtN =√N
        xl1 = new_vars[1]
        JuMP.@constraint(model, t <= xl1 / rtN)

        function _getx(i)
            if i > n
                # if we are beyond the number of variables in the actual geometric mean, we are adding the buffer variable
                return xl1 / √N # bounds hypograph variable
            else
                # otherwise we are adding the (i+1)th variable
                return end_pts[i]
            end
        end

        offset = offset_next = 0
        # loop over layers, layer 1 describes hypograph variable
        for i in 1:l
            offset_next = offset + 2 ^ (i - 1) # **********************
            # loop over variables in each layer
            @show i
            for j in 1:(2 ^ (i - 1))
                @show j
                @show offset, offset_next
                @show offset + j, offset_next + 2j - 1, offset_next + 2j
                if i == l
                    # in the last layer, we use the original variables
                    a = _getx(2j - 1)
                    b = _getx(2j)
                else
                    a = new_vars[offset_next + 2j - 1]
                    b = new_vars[offset_next + 2j]
                end
                c = new_vars[offset + j]
                JuMP.@constraint(model, [a, b, c] in JuMP.RotatedSecondOrderCone())
            end
            offset = offset_next
        end

    else
        error("unknown cone $(constr_cone)")
    end

    println(model)

    return (model = model,)
end

maxvolumeJuMP1() = maxvolumeJuMP(4, constr_cone = :geomean)
maxvolumeJuMP2() = maxvolumeJuMP(4, constr_cone = :soc)

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
