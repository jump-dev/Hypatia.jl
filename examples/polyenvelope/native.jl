#=
formulates and solves the (dual of the) polynomial envelope problem described in:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using SparseArrays

struct PolyEnvelopeNative{T <: Real} <: ExampleInstanceNative{T}
    n::Int
    rand_halfdeg::Int
    num_polys::Int
    env_halfdeg::Int
    primal_wsos::Bool # use primal formulation, else use dual
end

function build(inst::PolyEnvelopeNative{T}) where {T <: Real}
    (n, num_polys) = (inst.n, inst.num_polys)
    @assert inst.rand_halfdeg <= inst.env_halfdeg
    domain = PolyUtils.BoxDomain{T}(-ones(T, n), ones(T, n))

    # generate interpolation
    (U, pts, Ps, _, w) = PolyUtils.interpolate(domain,
        inst.env_halfdeg, get_quadr = true)

    # generate random data
    L = binomial(n + inst.rand_halfdeg, n)
    c_or_h = vec(Ps[1][:, 1:L] * rand(T(-9):T(9), L, num_polys))

    if inst.primal_wsos
        # WSOS cone in primal
        c = -w
        A = zeros(T, 0, U)
        b = T[]
        G = repeat(sparse(one(T) * I, U, U), outer = (num_polys, 1))
        h = c_or_h
    else
        # WSOS cone in dual
        c = c_or_h
        A = repeat(sparse(one(T) * I, U, U), outer = (1, num_polys))
        b = w
        G = Diagonal(-one(T) * I, num_polys * U) # TODO uniformscaling
        h = zeros(T, num_polys * U)
    end

    cones = Cones.Cone{T}[Cones.WSOSInterpNonnegative{T, T}(U, Ps,
        use_dual = !inst.primal_wsos) for k in 1:num_polys]

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end
