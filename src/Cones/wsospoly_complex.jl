#=
Copyright 2018, Chris Coey and contributors
=#

mutable struct WSOSComplex <: Cone
    use_dual::Bool
    dim::Int
    sidedims::Vector{Int}
    initpoint::Vector{Float64}
    Mfuncs::Vector{Function}
    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    # diffres

    function WSOSComplex(dim::Int, sidedims::Vector{Int}, initpoint::Vector{Float64}, Mfuncs::Vector{Function}, is_dual::Bool)
        cone = new()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.sidedims = sidedims
        cone.initpoint = initpoint
        cone.Mfuncs = Mfuncs
        cone.g = Vector{Float64}(undef, dim)
        cone.H = Matrix{Float64}(undef, dim, dim)
        cone.H2 = similar(cone.H)
        # realify(X) = Symmetric([real.(X) -imag.(X)'; imag.(X) real.(X)], :L)
        # barfun(point) = -sum(logdet(cholesky(realify(f(point)))) for f in Mfuncs)
        barfun(point) = -sum(logdetchol(f(point)) for f in Mfuncs)
        cone.barfun = barfun
        # cone.diffres = DiffResults.HessianResult(cone.g)
        return cone
    end
end

WSOSComplex(dim::Int, sidedims::Vector{Int}, initpoint::Vector{Float64}, Mfuncs::Vector{Function}) = WSOSComplex(dim, sidedims, initpoint, Mfuncs, false)

# get_nu(cone::WSOSComplex) = 2 * sum(cone.sidedims)
get_nu(cone::WSOSComplex) = sum(cone.sidedims)

set_initial_point(arr::AbstractVector{Float64}, cone::WSOSComplex) = (@. arr = cone.initpoint; arr)

function check_in_cone(cone::WSOSComplex)
    for f in cone.Mfuncs
        mat = f(cone.point)
        if !isposdef(mat)
            # @show mat
            return false
        end
    end

    # # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    # cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    # cone.g .= DiffResults.gradient(cone.diffres)
    # cone.H .= DiffResults.hessian(cone.diffres)

    return factorize_hess(cone)
end
