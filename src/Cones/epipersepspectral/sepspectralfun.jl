#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
suitable univariate convex functions defined on ℝ₊₊ and associated oracles
- h_val evaluates (sum of) h
- h_conj_dom_pos is true if domain of conjugate of h is ℝ₊₊, else domain is ℝ
- h_conj evaluates the convex conjugate of (sum of) h
- h_der1/h_der2/h_der3 evaluate 1st/2nd/3rd derivatives of h pointwise
- get_initial_point gives initial point for u,v,λ_i

TODO derive central initial points for get_initial_point and use real type
=#

"""
$(TYPEDEF)

The negative logarithm function ``x \\to - \\log(x)``.
"""
struct NegLogSSF <: SepSpectralFun end

h_val(xs::Vector{T}, ::NegLogSSF) where {T <: Real} = -sum(log, xs)

h_conj_dom_pos(::NegLogSSF) = true

h_conj(xs::Vector{T}, ::NegLogSSF) where {T <: Real} = -length(xs) - sum(log, xs)

h_der1(ds::Vector{T}, xs::Vector{T}, ::NegLogSSF) where {T <: Real} = (@. ds = -inv(xs))
h_der2(ds::Vector{T}, xs::Vector{T}, ::NegLogSSF) where {T <: Real} = (@. ds = xs^-2)
h_der3(ds::Vector{T}, xs::Vector{T}, ::NegLogSSF) where {T <: Real} = (@. ds = -2 * xs^-3)

function get_initial_point(d::Int, ::NegLogSSF)
    # TODO initial central point
    return (1, 1, 1)
end

"""
$(TYPEDEF)

The negative entropy function ``x \\to x \\log(x)``.
"""
struct NegEntropySSF <: SepSpectralFun end

h_val(xs::Vector{T}, ::NegEntropySSF) where {T <: Real} = sum(x * log(x) for x in xs)

h_conj_dom_pos(::NegEntropySSF) = false

h_conj(xs::Vector{T}, ::NegEntropySSF) where {T <: Real} = sum(exp(-x - 1) for x in xs)

function h_der1(ds::Vector{T}, xs::Vector{T}, ::NegEntropySSF) where {T <: Real}
    return (@. ds = 1 + log(xs))
end
h_der2(ds::Vector{T}, xs::Vector{T}, ::NegEntropySSF) where {T <: Real} = (@. ds = inv(xs))
h_der3(ds::Vector{T}, xs::Vector{T}, ::NegEntropySSF) where {T <: Real} = (@. ds = -xs^-2)

function get_initial_point(d::Int, ::NegEntropySSF)
    # TODO initial central point
    return (1, 1, 1)
end

"""
$(TYPEDEF)

The negative square root function ``x \\to -x^{1/2}``. Note this is a special
case of the negative power: `NegPower01SSF(0.5)`.
"""
struct NegSqrtSSF <: SepSpectralFun end

h_val(xs::Vector{T}, ::NegSqrtSSF) where {T <: Real} = -sum(sqrt, xs)

h_conj_dom_pos(::NegSqrtSSF) = true

h_conj(xs::Vector{T}, ::NegSqrtSSF) where {T <: Real} = T(0.25) * sum(inv, xs)

function h_der1(ds::Vector{T}, xs::Vector{T}, ::NegSqrtSSF) where {T <: Real}
    return (@. ds = T(-0.5) * inv(sqrt(xs)))
end
function h_der2(ds::Vector{T}, xs::Vector{T}, ::NegSqrtSSF) where {T <: Real}
    return (@. ds = T(0.25) * xs^T(-1.5))
end
function h_der3(ds::Vector{T}, xs::Vector{T}, ::NegSqrtSSF) where {T <: Real}
    return (@. ds = T(-3 / 8) * xs^T(-2.5))
end

function get_initial_point(d::Int, ::NegSqrtSSF)
    # TODO initial central point
    return (0, 1, 1)
end

"""
$(TYPEDEF)

The negative power function ``x \\to -x^p`` parametrized by ``p \\in (0, 1)``
"""
struct NegPower01SSF <: SepSpectralFun
    p::Real
    NegPower01SSF(p::Real) = (@assert 0 < p < 1; new(p))
end

h_val(xs::Vector{T}, h::NegPower01SSF) where {T <: Real} = -sum(x^T(h.p) for x in xs)

h_conj_dom_pos(::NegPower01SSF) = true

# -(p - 1) * sum((x / p)^q for x in xs)
function h_conj(xs::Vector{T}, h::NegPower01SSF) where {T <: Real}
    p = T(h.p)
    q = p / (p - 1)
    return (1 - p) * p^-q * sum(x^q for x in xs)
end

function h_der1(ds::Vector{T}, xs::Vector{T}, h::NegPower01SSF) where {T <: Real}
    p = T(h.p)
    pm1 = p - 1
    @. ds = -p * xs^pm1
    return ds
end
function h_der2(ds::Vector{T}, xs::Vector{T}, h::NegPower01SSF) where {T <: Real}
    p = T(h.p)
    pm2 = p - 2
    coef = -p * (p - 1)
    @. ds = coef * xs^pm2
    return ds
end
function h_der3(ds::Vector{T}, xs::Vector{T}, h::NegPower01SSF) where {T <: Real}
    p = T(h.p)
    pm3 = p - 3
    coef = -p * (p - 1) * (p - 2)
    @. ds = coef * xs^pm3
    return ds
end

function get_initial_point(d::Int, h::NegPower01SSF)
    # TODO initial central point
    return (0, 1, 1)
end

"""
$(TYPEDEF)

The power function ``x \\to x^p`` parametrized by ``p \\in (1, 2]``. Note for
``p = 2``, it is more efficient to use [`EpiPerSquare`](@ref).
"""
struct Power12SSF <: SepSpectralFun
    p::Real
    Power12SSF(p::Real) = (@assert 1 < p <= 2; new(p))
end

h_val(xs::Vector{T}, h::Power12SSF) where {T <: Real} = sum(x^T(h.p) for x in xs)

h_conj_dom_pos(::Power12SSF) = false

# (p - 1) * sum((x >= 0 ? zero(x) : (abs(x) / p)^q) for x in xs)
function h_conj(xs::Vector{T}, h::Power12SSF) where {T <: Real}
    p = T(h.p)
    q = p / (p - 1)
    val = zero(T)
    for x in xs
        if x < 0
            val += abs(x)^q
        end
    end
    return (p - 1) * p^-q * val
end

function h_der1(ds::Vector{T}, xs::Vector{T}, h::Power12SSF) where {T <: Real}
    p = T(h.p)
    pm1 = p - 1
    @. ds = p * xs^pm1
    return ds
end
function h_der2(ds::Vector{T}, xs::Vector{T}, h::Power12SSF) where {T <: Real}
    p = T(h.p)
    pm2 = p - 2
    coef = p * (p - 1)
    @. ds = coef * xs^pm2
    return ds
end
function h_der3(ds::Vector{T}, xs::Vector{T}, h::Power12SSF) where {T <: Real}
    p = T(h.p)
    pm3 = p - 3
    coef = p * (p - 1) * (p - 2)
    @. ds = coef * xs^pm3
    return ds
end

function get_initial_point(d::Int, h::Power12SSF)
    # TODO initial central point
    return (2d, 1, 1)
end
