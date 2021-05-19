#=
lower bound on relative entropy of entanglement (PPT relaxation)
adapted from
https://github.com/hfawzi/cvxquad/blob/master/examples/rel_entr_entanglement.m
=#

struct RelEntrEntanglementJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    na::Int
    nb::Int
end

function build(inst::RelEntrEntanglementJuMP{T}) where {T <: Float64}
    (na, nb) = (inst.na, inst.nb)
    side = na * nb
    Rho = randn(T, side, side)
    Rho = Rho * Rho'
    Rho = Symmetric(Rho / tr(Rho))
    vec_dim = Cones.svec_length(side)
    rho_vec = zeros(T, vec_dim)
    Cones.smat_to_svec!(rho_vec, Rho, sqrt(T(2)))

    model = JuMP.Model()
    JuMP.@variable(model, tau_vec[1:vec_dim])
    Tau = zeros(JuMP.AffExpr, side, side)
    Cones.svec_to_smat!(Tau, one(T) * tau_vec, sqrt(T(2)))
    JuMP.@constraint(model, tr(Tau) == 1)

    JuMP.@variable(model, y)
    JuMP.@objective(model, Min, y / log(T(2)))
    JuMP.@constraint(model, vcat(y, tau_vec, rho_vec) in
        Hypatia.EpiTrRelEntropyTriCone{T}(1 + 2 * vec_dim))
    pt = partial_transpose(Symmetric(Tau), 2, [na, nb])
    JuMP.@SDconstraint(model, Symmetric(pt) >= 0)

    return model
end

# partial transpose of Q over system i given subsystem dimensions subs
function partial_transpose(
    Q::Symmetric,
    i::Int,
    subs::Vector{Int},
    )
    @assert 1 <= i <= length(subs)
    @assert size(Q, 1) == prod(subs)
    n = length(subs)
    s = n + 1 - i
    p = collect(1:2n)
    p[s] = n + s
    p[n + s] = s
    rev = reverse(subs)
    permQ = permutedims(reshape(Q, (rev..., rev...)), p)
    side = prod(subs)
    return reshape(permQ, (side, side))
end
