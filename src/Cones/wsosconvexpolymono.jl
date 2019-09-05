#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#

using DynamicPolynomials, ForwardDiff, DiffResults

mutable struct WSOSConvexPolyMonomial{T <: Real} <: Cone{T}
    use_dual::Bool
    n::Int
    deg::Int
    dim::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    poly_pairs
    get_lambda
    barfun
    diffres

    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact
    hess_fact_cache

    function WSOSConvexPolyMonomial{T}(n::Int, deg::Int, is_dual::Bool) where {T <: Real}
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        cone.n = n
        cone.deg = deg
        dim = binomial(cone.n + deg, cone.n) - binomial(cone.n + 1, cone.n)
        @show cone.dim
        cone.dim = dim
        return cone
    end
end

WSOSConvexPolyMonomial{T}(n::Int, deg::Int) where {T <: Real} = WSOSConvexPolyMonomial{T}(n::Int, deg::Int, false)

function setup_data(cone::WSOSConvexPolyMonomial{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)

    # create a lookup table of coefficients we will use to build lambda later
    n = cone.n
    @polyvar x[1:n]
    monos_sqr = monomials(x, 2:cone.deg)
    monos_hess = monomials(x, 0:cone.deg - 2)
    monos_low = monomials(x, 0:div(cone.deg - 2, 2))
    L = length(monos_low)
    poly_pairs = [Float64[] for i in 1:L, j in 1:L]
    for i in 1:length(monos_low), j in 1:i, m in monos_hess
        poly = monos_low[i] * monos_low[j]
        push!(poly_pairs[i, j], coefficient(poly, m))
    end
    cone.poly_pairs = poly_pairs

    lifting = zeros(length(monos_hess) * div(n * (n + 1), 2), length(monos_sqr))
    for k in 1:length(monos_sqr)
        basis_poly = monos_sqr[k]
        hess = differentiate(basis_poly, x, 2)
        lifting[:, k] = vcat([coefficient(hess[i, j], m) for i in 1:n for j in 1:i for m in monos_hess]...)
    end

    integrating = similar(lifting)
    for i in 1:size(lifting, 1), j in 1:size(lifting, 2)
        aij = lifting[i, j]
        integrating[i, j] = iszero(aij) ? 0 : inv(aij)
    end

    scalevals = diag(lifting' * integrating)
    integrating ./= scalevals'

    function get_lambda(point)
        L = length(monos_low)
        lambda = zeros(eltype(point), L * n, L * n)
        for (u, msu) in enumerate(monos_sqr)
            idxs = []
            num_repeats = 0
            for i in 1:n, j in 1:i
                di = degree(msu, x[i])
                dj = degree(msu, x[j])
                if (i == j && di >= 2) || (i != j && di >= 1 && dj >= 1)
                    num_repeats += 1
                else
                    continue
                end
                for k in 1:length(monos_low), l in 1:k
                    if msu != monos_low[k] * monos_low[l] * x[i] * x[j]
                        continue
                    end
                    if i == j
                        fact = inv(di * (di - 1))
                    else
                        fact = inv(di * dj)
                    end
                    lambda[(i - 1) * L + k, (j - 1) * L + l] = lambda[(i - 1) * L + l, (j - 1) * L + k] = fact * point[u]
                    push!(idxs, (i, j, k, l))
                end # inner
            end # outer
            # weights = rand(num_repeats)
            # weights ./= sum(weights)
            # @show (weights)
            # @show msu, weights
            # if msu == x[1] ^ 2 * x[2] ^ 2
            #     weights = [0.5, 0, 0.5]
            # elseif msu == x[1] ^ 3 * x[2]
            #     weights = [0, 1]
            # elseif msu == x[2] ^ 3 * x[1]
            #     weights = [0, 1]
            # else
                weights = fill(1 / num_repeats, num_repeats)
            # end
            for (w, (i, j, k, l)) in enumerate(idxs)
                lambda[(i - 1) * L + k, (j - 1) * L + l] *= weights[w]
                if k != l
                    lambda[(i - 1) * L + l, (j - 1) * L + k] *= weights[w]
                end
            end
        end # monos sqr
        return lambda
    end

    # function get_lambda(point)
    #     hess_fullspace = integrating * point
    #     lambda = zeros(eltype(point), n * L, n * L)
    #     U = length(monos_hess)
    #     u = 1
    #     for i in 1:n, j in 1:i
    #         point_coeffs = view(hess_fullspace, u:(u + U - 1))
    #         for k in 1:L, l in 1:k
    #             lambda[(i - 1) * L + k, (j - 1) * L + l] = lambda[(i - 1) * L + l, (j - 1) * L + k] = dot(poly_pairs[k, l], point_coeffs)
    #         end
    #         u += U
    #     end
    #     return lambda
    # end
    cone.get_lambda = get_lambda

    function barfun(point)
        return -logdet(Symmetric(get_lambda(point), :L))
    end
    cone.barfun = barfun
    cone.diffres = DiffResults.HessianResult(cone.grad)

    cone.hess_fact_cache = nothing
    return
end

get_nu(cone::WSOSConvexPolyMonomial) = binomial(cone.n + div(cone.deg - 2, 2), cone.n) * cone.n

function set_initial_point(arr::AbstractVector, cone::WSOSConvexPolyMonomial)
    @assert cone.deg == 4
    arr .= 0
    if cone.n == 2
        arr[1] = 1
        arr[3] = 1
        arr[5] = 1.01 # *
        arr[10] = 1
        arr[12] = 1
    elseif cone.n == 3
        # arr[1] = 1.2
        # arr[4] = 10
        # arr[6] = 10
        # arr[11] = 1.2
        # arr[13] = 10
        # arr[15] = 1.2
        # arr[26] = 0.2
        # arr[29] = 0.2
        # arr[31] = 0.2
        # @show arr
        arr .= [
        13.96770224404994
         0
         0.0
         4.933853779793987
         0.0
         4.933853779463825
         0.0
         0.0
         0.0
         0
        13.967702240748276
         0.0
         4.933853779133654
         0.0
        13.967702244049939
         0.0
         0.0
         0.0
         0.0
         0.0
         0.0
         0.0
         0.0
         0.0
         0
         2
         0.0
         0.0
         2.0
         0.0
         2
          ]
    end
    return arr
end

function update_feas(cone::WSOSConvexPolyMonomial)
    @assert !cone.feas_updated
    cone.is_feas = isposdef(Symmetric(cone.get_lambda(cone.point), :L))
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSConvexPolyMonomial)
    @assert cone.is_feas
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.grad .= DiffResults.gradient(cone.diffres)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSConvexPolyMonomial)
    @assert cone.grad_updated
    cone.hess.data .= DiffResults.hessian(cone.diffres)
    cone.hess_updated = true
    return cone.hess
end

# function inv_hess(cone::WSOSConvexPolyMonomial)
#     cone.inv_hess = inv(cone.hess)
#     cone.inv_hess_updated = true
#     return cone.inv_hess
# end
