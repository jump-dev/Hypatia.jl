#=
references:
- https://github.com/hfawzi/cvxquad/blob/master/examples/entanglement_assisted_capacity.m
- listing 2 in "Efficient optimization of the quantum relative entropy" by H. Fawzi and O. Fawzi
=#
function build(inst::EntanglementAssisted{T}) where {T <: Float64}
    gamma = 0.2
    ampl_damp = [1 0; 0 sqrt(gamma); 0 sqrt(1-gamma); 0 0]
    na = 2
    nb = 2
    ne = 2
    rt2 = sqrt(2)
    sn = div(na * (na + 1), 2)

    model = JuMP.Model()
    JuMP.@variables(model, begin
        ρ[1:na, 1:na], PSD
        cond_epi
        qe_epi
    end)

    Q1 = ampl_damp * ρ * ampl_damp'
    Q2 = kron(I, ptrace(Q1, [nb, ne], 1)) # what?
    Q3 = ptrace(Q1, [nb, ne], 2)

    JuMP.@constraints(model, begin
        vcat(cond_epi, Cones.smat_to_svec!(zeros(JuMP.AffExpr, sn), Q1, rt2), Cones.smat_to_svec!(zeros(JuMP.AffExpr, sn), Q2, rt2)) in Hypatia.EpiTraceRelEntropyTriCone{Float64}(1 + 2 * sn)
        vcat(qe_epi, Cones.smat_to_svec!(zeros(JuMP.AffExpr, sn), Q3, rt2)) in Hypatia.EpiPerTraceEntropyTriCone{Float64}(1 + sn)
        tr(ρ) == 1
    end)

    JuMP.@objective(model, Max, (cond_epi + qe_epi) / log(2))

    return model
end
