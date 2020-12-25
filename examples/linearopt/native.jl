#=
solves a simple random linear optimization problem (LP):
    min  c'x
    s.t. Ax = b, x >= 0
=#

using SparseArrays

struct LinearOptNative{T <: Real} <: ExampleInstanceNative{T}
    m::Int
    n::Int
    nz_frac::Float64
end

function build(inst::LinearOptNative{T}) where {T <: Real}
    (m, n, nz_frac) = (inst.m, inst.n, inst.nz_frac)
    @assert 0 < nz_frac <= 1

    # generate random data
    A = (nz_frac >= 1) ? rand(T, m, n) : sprand(T, m, n, nz_frac)
    A .*= 10
    b = vec(sum(A, dims = 2))
    c = rand(T, n)
    G = Diagonal(-one(T) * I, n) # TODO uniformscaling
    h = zeros(T, n)
    cones = Cones.Cone{T}[Cones.Nonnegative{T}(n)]

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end
