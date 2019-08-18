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

mutable struct EpiPerPower{T <: Real} <: Cone{T}
    use_dual::Bool
    alpha::T
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

    barfun::Function
    diffres
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact # TODO prealloc

    function EpiPerPower{T}(alpha::T, is_dual::Bool) where {T <: Real}
        @assert alpha > 1
        cone = new()
        cone.use_dual = is_dual
        cone.alpha = alpha
        ialpha2 = 2 / alpha
        if cone.alpha >= 2
            cone.barfun = (s -> -log(s[1] * s[2] ^ (2 - ialpha2) - abs2(s[3]) * s[1] ^ (1 - ialpha2)))
        else
            cone.barfun = (s -> -log(s[1] ^ ialpha2 * s[2] - abs2(s[3]) * s[2] ^ (ialpha2 - 1)))
        end
        return cone
    end
end

EpiPerPower{T}(alpha::T) where {T <: Real} = EpiPerPower{T}(alpha, false)

function setup_data(cone::EpiPerPower{T}) where {T <: Real}
    reset_data(cone)
    cone.point = zeros(T, 3)
    cone.grad = zeros(T, 3)
    cone.hess = Symmetric(zeros(T, 3, 3), :U)
    cone.tmp_hess = Symmetric(zeros(T, 3, 3), :U)
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
    cone.is_feas = (u > 0 && v > 0 && log(abs(w)) < log(u) / cone.alpha + log(v) * (cone.alpha - 1) / cone.alpha)
    cone.feas_updated = true
    return cone.is_feas
end
