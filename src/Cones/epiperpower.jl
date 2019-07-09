#=
Copyright 2018, Chris Coey and contributors

epigraph of perspective of convex power of absolute value function (AKA 3-dim power cone) parametrized by real alpha > 1
(u in R, v in R_+, w in R) : u >= v*|w/v|^alpha
equivalent to u >= v^(1-alpha)*|w|^alpha or u^(1/alpha)*v^(1-1/alpha) >= |w|

barrier from "Cones and Interior-Point Algorithms for Structured Convex Optimization involving Powers and Exponentials" by P. Chares 2007
-log(u^(2/alpha)*v^(2-2/alpha) - w^2) - max{1-2/alpha, 0}*log(u) - max{2/alpha-1, 0}*log(v)

TODO get gradient and hessian analytically (may be nicer if redefine as u >= v/alpha*|w/v|^alpha)
TODO although this barrier has a lower parameter, maybe the more standard barrier is more numerically robust
=#

mutable struct EpiPerPower{T <: HypReal} <: Cone{T}
    use_dual::Bool
    alpha::Real
    point::AbstractVector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    barfun::Function
    diffres
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact # TODO prealloc

    function EpiPerPower{T}(alpha::Real, is_dual::Bool) where {T <: HypReal}
        @assert alpha > 1.0
        cone = new()
        cone.use_dual = is_dual
        cone.alpha = alpha
        return cone
    end
end

EpiPerPower{T}(alpha::Real) where {T <: HypReal} = EpiPerPower{T}(alpha, false)

reset_data(cone::EpiPerPower) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_prod_updated = false)

function setup_data(cone::EpiPerPower{T}) where {T <: HypReal}
    reset_data(cone)
    cone.grad = zeros(T, 3)
    cone.hess = Symmetric(zeros(T, 3, 3), :U)
    cone.tmp_hess = Symmetric(zeros(T, 3, 3), :U)
    ialpha2 = 2 / cone.alpha
    if cone.alpha >= 2
        cone.barfun = point -> -log(point[1] * point[2]^(2 - ialpha2) - abs2(point[3]) * point[1]^(1 - ialpha2))
    else
        cone.barfun = point -> -log(point[1]^ialpha2 * point[2] - abs2(point[3]) * point[2]^(ialpha2 - 1))
    end
    cone.diffres = DiffResults.HessianResult(cone.grad)
    return
end

dimension(cone::EpiPerPower) = 3

get_nu(cone::EpiPerPower) = 3 - 2 * min(inv(cone.alpha), 1 - inv(cone.alpha))

function set_initial_point(arr::AbstractVector, cone::EpiPerPower)
    arr[1] = 1
    arr[2] = 1
    arr[3] = 0
    return arr
end

function update_feas(cone::EpiPerPower)
    @assert !cone.feas_updated
    (u, v, w) = cone.point
    cone.is_feas = u > 0 && v > 0 && u > v * (abs(w / v))^cone.alpha
    cone.feas_updated = true
    return cone.is_feas
end

# TODO check if this is most efficient way to use DiffResults
function update_grad(cone::EpiPerPower)
    @assert cone.is_feas
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.grad .= DiffResults.gradient(cone.diffres)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiPerPower)
    @assert cone.grad_updated
    cone.hess.data .= DiffResults.hessian(cone.diffres)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess_prod(cone::EpiPerPower)
    @assert cone.hess_updated
    copyto!(cone.tmp_hess, cone.hess)
    cone.hess_fact = hyp_chol!(cone.tmp_hess)
    cone.inv_hess_prod_updated = true
    return
end

function update_inv_hess(cone::EpiPerPower)
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    cone.inv_hess = Symmetric(inv(cone.hess_fact), :U)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

# TODO maybe write using linear operator form rather than needing explicit hess
function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerPower)
    @assert cone.hess_updated
    return mul!(prod, cone.hess, arr)
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerPower)
    @assert cone.inv_hess_prod_updated
    return ldiv!(prod, cone.hess_fact, arr)
end
