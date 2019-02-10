#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
Copyright 2018, David Papp, Sercan Yildiz



definition and dual barrier extended from from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

# Affine transform of a function in the wsos cone. Rather than this being its own cone could adapt functions from here to act upon other cones.

mutable struct MonotonicPoly <: Cone # not properly named
    use_dual::Bool
    dim::Int
    ipwt::Vector{Matrix{Float64}}
    transform::Vector{Matrix{Float64}} # can be an operator
    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F # TODO prealloc
    tmp1::Vector{Matrix{Float64}}
    tmp2::Vector{Matrix{Float64}}
    tmp3::Matrix{Float64}

    function MonotonicPoly(dim::Int, ipwt::Vector{Matrix{Float64}}, transform::Vector{Matrix{Float64}}, is_dual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == dim
        end
        for tr in transform
            @assert size(tr) == (dim, dim)
            # @assert transform invertible, transforms will need to be consistent by assumption
        end
        cone = new()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.ipwt = ipwt
        cone.transform = transform
        cone.g = similar(ipwt[1], dim)
        cone.H = similar(ipwt[1], dim, dim)
        cone.H2 = similar(cone.H)
        cone.tmp1 = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        cone.tmp2 = [similar(ipwt[1], size(ipwtj, 2), dim) for ipwtj in ipwt]
        cone.tmp3 = similar(ipwt[1], dim, dim)
        return cone
    end
end

MonotonicPoly(dim::Int, ipwt::Vector{Matrix{Float64}}, transform::Vector{Matrix{Float64}}) = MonotonicPoly(dim, ipwt, transform, false)

get_nu(cone::MonotonicPoly) = sum(size(ipwtj, 2) for ipwtj in cone.ipwt)

set_initial_point(arr::AbstractVector{Float64}, cone::MonotonicPoly) = (@. arr = 1.0; arr .= cone.transform[1] \ arr; arr)

function check_in_cone(cone::MonotonicPoly)
    @. cone.g = 0.0
    @. cone.H = 0.0
    tmp3 = cone.tmp3
    transform = cone.transform

    for tr in transform, j in eachindex(cone.ipwt) # TODO can be done in parallel, but need multiple tmp3s
        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        tmp2j = cone.tmp2[j]

        # tmp1j = ipwtj'*Diagonal(point)*ipwtj
        # mul!(tmp2j, ipwtj', Diagonal(cone.point)) # TODO dispatches to an extremely inefficient method
        tmp2j .= ipwtj' .* (tr * cone.point)' # transform(cone.point)
        mul!(tmp1j, tmp2j, ipwtj)

        # pivoted cholesky and triangular solve method
        F = cholesky!(Symmetric(tmp1j, :L), Val(true), check = false)
        if !isposdef(F)
            return false
        end

        tmp2j .= view(ipwtj', F.p, :)
        ldiv!(F.L, tmp2j) # TODO make sure calls best triangular solve
        # mul!(tmp3, tmp2j', tmp2j)
        BLAS.syrk!('U', 'T', 1.0, tmp2j, 0.0, tmp3)

        @inbounds for i in eachindex(cone.g)
            cone.g[i] -= dot(diag(tmp3), tr[:, i])
            @inbounds for k in 1:i
                cone.H[k, i] += sum(abs2(tmp3[p, q]) * tr[p, k] * tr[q, i] for p in 1:cone.dim, q in 1:cone.dim)
            end
        end
    end

    return factorize_hess(cone)
end
