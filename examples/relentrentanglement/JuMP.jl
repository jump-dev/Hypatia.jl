#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
lower bound on relative entropy of entanglement (PPT relaxation)
adapted from
https://github.com/hfawzi/cvxquad/blob/master/examples/rel_entr_entanglement.m
=#

struct RelEntrEntanglementJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    na::Int
    nb::Int
end

function build(inst::RelEntrEntanglementJuMP{T}) where {T <: Real}
    (na, nb) = (inst.na, inst.nb)
    side = na * nb
    Rho = randn(Complex{T}, side, side)
    Rho = Rho * Rho'
    Rho = Hermitian(Rho / tr(Rho))
    vec_dim = Cones.svec_length(Complex, side)
    rho_vec = Vector{T}(undef, vec_dim)
    Cones.smat_to_svec!(rho_vec, Rho, sqrt(T(2)))

    model = JuMP.GenericModel{T}()
    JuMP.@variable(model, tau_vec[1:vec_dim])
    Tau = Matrix{JuMP.GenericAffExpr{Complex{T}, JuMP.GenericVariableRef{T}}}(
        undef,
        side,
        side,
    )
    Cones._svec_to_smat_complex!(Tau, one(T) * tau_vec, sqrt(T(2)))
    JuMP.@constraint(model, tr(Tau) == 1)

    JuMP.@variable(model, y)
    JuMP.@objective(model, Min, y / log(T(2)))
    JuMP.@constraint(
        model,
        vcat(y, tau_vec, rho_vec) in
        Hypatia.EpiTrRelEntropyTriCone{T, Complex{T}}(1 + 2 * vec_dim)
    )
    pt = partial_transpose(Hermitian(Tau), 2, [na, nb])
    JuMP.@constraint(model, Hermitian(pt) in JuMP.HermitianPSDCone())

    return model
end

# partial transpose of Q over system i given subsystem dimensions subs
function partial_transpose(Q::AbstractMatrix, i::Int, subs::Vector{Int})
    @assert 1 <= i <= length(subs)
    @assert size(Q, 1) == prod(subs)
    n = length(subs)
    s = n + 1 - i
    p = collect(1:(2n))
    p[s] = n + s
    p[n + s] = s
    rev = reverse(subs)
    permQ = permutedims(reshape(Q, (rev..., rev...)), p)
    side = prod(subs)
    return reshape(permQ, (side, side))
end
