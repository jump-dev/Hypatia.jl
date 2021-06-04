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

function build(inst::EntanglementAssisted{T}) where {T <: Float64}
    gamma = 0.2
    ampl_damp = [
        1 0;
        0 sqrt(gamma);
        0 sqrt(1 - gamma);
        0 0;
        ]
    ampl_dim = 4
    na = 2
    nb = inst.nb
    ne = inst.ne
    @assert nb * ne == ampl_dim
    rt2 = sqrt(T(2))
    sa = Cones.svec_length(ampl_dim)
    sb = Cones.svec_length(nb)

    model = JuMP.Model()
    JuMP.@variables(model, begin
        ρ[1:na, 1:na], PSD
        cond_epi
        qe_epi
    end)

    Q1 = Symmetric(ampl_damp * ρ * ampl_damp')
    Q2 = zeros(JuMP.AffExpr, nb * ne, nb * ne)
    kron!(Q2, I(nb), partial_trace(Q1, 1, [nb, ne]))
    Q3 = partial_trace(Q1, 2, [nb, ne])
    Q1_vec = Cones.smat_to_svec!(zeros(JuMP.AffExpr, sa), Q1, rt2)
    Q2_vec = Cones.smat_to_svec!(zeros(JuMP.AffExpr, sa), Q2, rt2)
    Q3_vec = Cones.smat_to_svec!(zeros(JuMP.AffExpr, sb), Q3, rt2)
    RE_cone = Hypatia.EpiTrRelEntropyTriCone{T}(1 + 2 * sa)
    E_cone = Hypatia.EpiPerSepSpectralCone{T}(Cones.NegEntropySSF(),
        Cones.MatrixCSqr{T, T}, nb)

    JuMP.@constraints(model, begin
        tr(ρ) == 1
        vcat(cond_epi, Q1_vec, Q2_vec) in RE_cone
        vcat(qe_epi, 1, Q3_vec) in E_cone
    end)

    JuMP.@objective(model, Max, (cond_epi + qe_epi) / -log(T(2)))

    return model
end

# partial trace of Q over system i given subsystem dimensions subs
function partial_trace(
    Q::Symmetric,
    i::Int,
    subs::Vector{Int},
    )
    @assert 1 <= i <= length(subs)
    @assert size(Q, 1) == prod(subs)
    return sum(partial_trace_j(j, Q, i, subs) for j in 1:subs[i])
end

function partial_trace_j(
    j::Int,
    Q::Symmetric,
    i::Int,
    subs::Vector{Int},
    )
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
