#=
utilities for complex polynomial interpolation

TODO make interface for complex and real interpolation consistent, use dispatch
=#

"""
$(SIGNATURES)

Compute interpolation data for a complex weighted sum-of-squares conic constraint
on a domain.
"""
function interpolate(
    R::Type{Complex{T}},
    halfdeg::Int,
    n::Int,
    gs::Vector,
    g_halfdegs::Vector{Int};
    sample_factor::Int = 10,
    use_qr::Bool = false,
    ) where {T <: Real}
    # generate interpolation
    # TODO use more numerically-stable basis for columns, and evaluate in a more
    # numerically stable way by multiplying the columns
    L = binomial(n + halfdeg, n)
    U = L^2
    L_basis = [a for t in 0:halfdeg for a in Combinatorics.multiexponents(n, t)]
    mon_pow(z, ex) = prod(z[i]^ex[i] for i in eachindex(ex))
    V_basis = [z -> mon_pow(z, L_basis[k]) * mon_pow(conj(z), L_basis[l]) for
        l in eachindex(L_basis) for k in eachindex(L_basis)]
    @assert length(V_basis) == U

    # sample from domain (inefficient for general domains, only samples from
    # unit box and checks feasibility)
    num_samples = sample_factor * U
    samples = Vector{Vector{Complex{T}}}(undef, num_samples)
    k = 0
    randbox() = 2 * rand(T) - 1
    while k < num_samples
        z = [Complex(randbox(), randbox()) for i in 1:n]
        if all(g -> g(z) > zero(T), gs)
            k += 1
            samples[k] = z
        end
    end

    # select subset of points to maximize abs(det(V)) in heuristic QR-based
    # procedure (analogous to real case)
    V = [b(z) for z in samples, b in V_basis]
    VF = qr(Matrix(transpose(V)), ColumnNorm())
    keep = VF.p[1:U]
    points = samples[keep]
    V = V[keep, :]

    # setup P matrices
    P0 = V[:, 1:L]
    if use_qr
        P0 = Matrix(qr(P0).Q)
    end
    Ps = [P0]
    for i in eachindex(gs)
        gi = gs[i].(points)
        @views P0i = P0[:, 1:binomial(n + halfdeg - g_halfdegs[i], n)]
        Pi = Diagonal(sqrt.(gi)) * P0i
        if use_qr
            Pi = Matrix(qr(Pi).Q)
        end
        push!(Ps, Pi)
    end

    return (points, Ps)
end
