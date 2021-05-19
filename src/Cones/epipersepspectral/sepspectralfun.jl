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

The inverse function ``x \\to x^{-1}``.
"""
struct InvSSF <: SepSpectralFun end

h_val(xs::Vector, ::InvSSF) = sum(inv, xs)
h_conj_dom_pos(::InvSSF) = true
h_conj(xs::Vector, ::InvSSF) = -2 * sum(sqrt, xs)

h_der1(ds::Vector, xs::Vector, ::InvSSF) = (@. ds = -xs^-2)
h_der2(ds::Vector, xs::Vector, ::InvSSF) = (@. ds = 2 * xs^-3)
h_der3(ds::Vector, xs::Vector, ::InvSSF) = (@. ds = -6 * xs^-4)

function get_initial_point(d::Int, ::InvSSF)
    # TODO initial central point
    return (2d, 1, 1)
end

"""
$(TYPEDEF)

The negative logarithm function ``x \\to - \\log(x)``.
"""
struct NegLogSSF <: SepSpectralFun end

h_val(xs::Vector, ::NegLogSSF) = -sum(log, xs)
h_conj_dom_pos(::NegLogSSF) = true
h_conj(xs::Vector, ::NegLogSSF) = -length(xs) - sum(log, xs)

h_der1(ds::Vector, xs::Vector, ::NegLogSSF) = (@. ds = -inv(xs))
h_der2(ds::Vector, xs::Vector, ::NegLogSSF) = (@. ds = xs^-2)
h_der3(ds::Vector, xs::Vector, ::NegLogSSF) = (@. ds = -2 * xs^-3)

function get_initial_point(d::Int, ::NegLogSSF)
    # TODO initial central point
    return (1, 1, 1)
end

"""
$(TYPEDEF)

The negative entropy function ``x \\to x \\log(x)``.
"""
struct NegEntropySSF <: SepSpectralFun end

h_val(xs::Vector, ::NegEntropySSF) = sum(x * log(x) for x in xs)
h_conj_dom_pos(::NegEntropySSF) = false
h_conj(xs::Vector, ::NegEntropySSF) = sum(exp(-x - 1) for x in xs)

h_der1(ds::Vector, xs::Vector, ::NegEntropySSF) = (@. ds = 1 + log(xs))
h_der2(ds::Vector, xs::Vector, ::NegEntropySSF) = (@. ds = inv(xs))
h_der3(ds::Vector, xs::Vector, ::NegEntropySSF) = (@. ds = -xs^-2)

function get_initial_point(d::Int, ::NegEntropySSF)
    # TODO initial central point
    return (1, 1, 1)
end

"""
$(TYPEDEF)

The power function ``x \\to x^p`` parametrized by ``p \\in (1, 2]``. Note for
``p = 2``, it is more efficient to use [`EpiPerSquare`](@ref) on the scaled
triangle.
"""
struct Power12SSF <: SepSpectralFun
    p::Real
    Power12SSF(p::Real) = (@assert 1 < p <= 2; new(p))
end

h_val(xs::Vector, h::Power12SSF) = sum(x^h.p for x in xs)
h_conj_dom_pos(::Power12SSF) = false
function h_conj(xs::Vector, h::Power12SSF)
    p = h.p
    q = p / (p - 1)
    return (p - 1) * sum((x >= 0 ? zero(x) : (abs(x) / p)^q) for x in xs)
end

function h_der1(ds::Vector, xs::Vector, h::Power12SSF)
    p = h.p
    pm1 = p - 1
    @. ds = p * xs^pm1
    return ds
end
function h_der2(ds::Vector, xs::Vector, h::Power12SSF)
    p = h.p
    pm2 = p - 2
    coef = p * (p - 1)
    @. ds = coef * xs^pm2
    return ds
end
function h_der3(ds::Vector, xs::Vector, h::Power12SSF)
    p = h.p
    pm3 = p - 3
    coef = p * (p - 1) * (p - 2)
    @. ds = coef * xs^pm3
    return ds
end

function get_initial_point(d::Int, h::Power12SSF)
    # TODO initial central point
    return (2d, 1, 1)
end
