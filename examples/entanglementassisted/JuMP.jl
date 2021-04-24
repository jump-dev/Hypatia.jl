#=
references:
- https://github.com/hfawzi/cvxquad/blob/master/examples/entanglement_assisted_capacity.m
- listing 2 in "Efficient optimization of the quantum relative entropy" by H. Fawzi and O. Fawzi

TODO waiting on https://github.com/Jutho/TensorOperations.jl/issues/77#issuecomment-818250083
=#
import QuantumInformation.ptrace
import QuantumInformation.tensortrace
function ptrace(ρ::AbstractMatrix, idims::Vector{Int}, isystems::Vector{Int})
    dims = reverse(idims)
    systems = length(idims) .- isystems .+ 1

    if size(ρ,1) != size(ρ,2)
        throw(ArgumentError("Non square matrix passed to ptrace"))
    end
    if prod(dims)!=size(ρ,1)
        throw(ArgumentError("Product of dimensions do not match shape of matrix."))
    end
    if maximum(systems) > length(dims) || minimum(systems) < 1
        throw(ArgumentError("System index out of range"))
    end
    offset = length(dims)
    keep = setdiff(1:offset, systems)

    traceidx = [1:offset; 1:offset]
    traceidx[keep] .+= offset

    tensor = reshape(ρ, [dims; dims]...)
    keepdim = prod([size(tensor, x) for x in keep])
    return reshape(tensortrace(tensor, Tuple(traceidx)), keepdim, keepdim)
end

struct EntanglementAssisted{T <: Real} <: ExampleInstanceJuMP{T}
    nb::Int
    ne::Int
end

function build(inst::EntanglementAssisted{T}) where {T <: Float64}
    gamma = 0.2
    ampl_damp = [1 0; 0 sqrt(gamma); 0 sqrt(1-gamma); 0 0]
    ampl_dim = 4
    na = 2
    nb = inst.nb
    ne = inst.ne
    @assert nb * ne == ampl_dim
    rt2 = sqrt(2)
    sa = div(ampl_dim * (ampl_dim + 1), 2)
    sb = div(nb * (nb + 1), 2)

    model = JuMP.Model()
    JuMP.@variables(model, begin
        ρ[1:na, 1:na], PSD
        cond_epi
        qe_epi
    end)

    Q1 = ampl_damp * ρ * ampl_damp'
    Q2 = kron!(zeros(JuMP.AffExpr, nb * ne, nb * ne), I(nb), ptrace(Q1, [nb, ne], [1]))
    Q3 = ptrace(Q1, [nb, ne], [2])

    JuMP.@constraints(model, begin
        vcat(cond_epi, Cones.smat_to_svec!(zeros(JuMP.AffExpr, sa), Q1, rt2), Cones.smat_to_svec!(zeros(JuMP.AffExpr, sa), Q2, rt2)) in Hypatia.EpiTraceRelEntropyTriCone{Float64}(1 + 2 * sa)
        vcat(qe_epi, 1, Cones.smat_to_svec!(zeros(JuMP.AffExpr, sb), Q3, rt2)) in Hypatia.EpiPerTraceEntropyTriCone{Float64}(2 + sb)
        tr(ρ) == 1
    end)

    JuMP.@objective(model, Max, -(cond_epi + qe_epi) / log(2))

    return model
end
