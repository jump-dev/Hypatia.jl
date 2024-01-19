#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
see:
https://github.com/hfawzi/cvxquad/blob/master/examples/entanglement_assisted_capacity.m
listing 2 in "Efficient optimization of the quantum relative entropy"
by H. Fawzi and O. Fawzi
=#

using SparseArrays

struct EntanglementAssisted{T <: Real} <: ExampleInstanceJuMP{T}
    nb::Int
    ne::Int
end

function build(inst::EntanglementAssisted{T}) where {T <: Real}
    gamma = T(1) / 5
    ampl_damp = [
        1 0
        0 sqrt(gamma)
        0 sqrt(1 - gamma)
        0 0
    ]
    ampl_dim = 4
    na = 2
    nb = inst.nb
    ne = inst.ne
    @assert nb * ne == ampl_dim
    rt2 = sqrt(T(2))
    sbe = Cones.svec_length(Complex, ampl_dim)
    sb = Cones.svec_length(Complex, nb)

    model = JuMP.GenericModel{T}()
    JuMP.@variables(model, begin
        ρA[1:na, 1:na], Hermitian
        conditional
        von_neumann
    end)

    ρBE = Hermitian(ampl_damp * ρA * ampl_damp')
    IρE = Matrix{JuMP.GenericAffExpr{Complex{T}, JuMP.GenericVariableRef{T}}}(
        undef,
        nb * ne,
        nb * ne,
    )
    kron!(IρE, I(nb), partial_trace(ρBE, 1, [nb, ne]))
    ρB = partial_trace(ρBE, 2, [nb, ne])

    ρBE_vec = Vector{JuMP.GenericAffExpr{T, JuMP.GenericVariableRef{T}}}(undef, sbe)
    IρE_vec = Vector{JuMP.GenericAffExpr{T, JuMP.GenericVariableRef{T}}}(undef, sbe)
    ρB_vec = Vector{JuMP.GenericAffExpr{T, JuMP.GenericVariableRef{T}}}(undef, sb)

    Cones._smat_to_svec_complex!(ρBE_vec, ρBE, rt2)
    Cones._smat_to_svec_complex!(IρE_vec, IρE, rt2)
    Cones._smat_to_svec_complex!(ρB_vec, ρB, rt2)
    RE_cone = Hypatia.EpiTrRelEntropyTriCone{T, Complex{T}}(1 + 2 * sbe)
    E_cone = Hypatia.EpiPerSepSpectralCone{T}(
        Cones.NegEntropySSF(),
        Cones.MatrixCSqr{T, Complex{T}},
        nb,
    )
    JuMP.@constraints(model, begin
        tr(ρA) == 1
        vcat(conditional, IρE_vec, ρBE_vec) in RE_cone
        vcat(von_neumann, 1, ρB_vec) in E_cone
    end)

    JuMP.@objective(model, Max, (conditional + von_neumann) / -log(T(2)))

    return model
end

# partial trace of Q over system i given subsystem dimensions subs
function partial_trace(Q::AbstractMatrix, i::Int, subs::Vector{Int})
    @assert 1 <= i <= length(subs)
    @assert size(Q, 1) == prod(subs)
    return sum(partial_trace_j(j, Q, i, subs) for j in 1:subs[i])
end

function partial_trace_j(j::Int, Q::AbstractMatrix, i::Int, subs::Vector{Int})
    X1 = sparse(I, 1, 1)
    X2 = copy(X1)
    for (k, dim) in enumerate(subs)
        if k == i
            spej = spzeros(Bool, dim, 1)
            spej[j] = 1
            X1 = kron(X1, spej')
            X2 = kron(X2, spej)
        else
            spI = sparse(I, dim, dim)
            X1 = kron(X1, spI)
            X2 = kron(X2, spI)
        end
    end
    return X1 * Q * X2
end
