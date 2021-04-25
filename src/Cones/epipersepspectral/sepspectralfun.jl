#=
suitable univariate functions defined on ℝ₊₊ with matrix monotone derivative, and associated oracles
=#

# negative logarithm: -log(x)
struct NegLogMMF <: SepSpectralFun end

h_val(xs::Vector, ::NegLogMMF) = -sum(log, xs)
h_conj_dom(xs::Vector, ::NegLogMMF) = all(>(eps(eltype(xs))), xs)
h_conj(xs::Vector, ::NegLogMMF) = -length(xs) - sum(log, xs)

h_der1(ds::Vector, xs::Vector, ::NegLogMMF) = (@. ds = -inv(xs))
h_der2(ds::Vector, xs::Vector, ::NegLogMMF) = (@. ds = xs^-2)
h_der3(ds::Vector, xs::Vector, ::NegLogMMF) = (@. ds = -2 * xs^-3)

function get_initial_point(d::Int, ::NegLogMMF)
    # @warn("not initial central point") # TODO
    return (1, 1, 1)
end


# negative entropy: x * log(x)
struct NegEntropyMMF <: SepSpectralFun end

h_val(xs::Vector, ::NegEntropyMMF) = sum(x * log(x) for x in xs)
h_conj_dom(xs::Vector, ::NegEntropyMMF) = true
h_conj(xs::Vector, ::NegEntropyMMF) = sum(exp(-x - 1) for x in xs)

h_der1(ds::Vector, xs::Vector, ::NegEntropyMMF) = (@. ds = 1 + log(xs))
h_der2(ds::Vector, xs::Vector, ::NegEntropyMMF) = (@. ds = inv(xs))
h_der3(ds::Vector, xs::Vector, ::NegEntropyMMF) = (@. ds = -xs^-2)

function get_initial_point(d::Int, ::NegEntropyMMF)
    # @warn("not initial central point") # TODO
    return (1, 1, 1)
end


# square: x^2
struct SquareMMF <: SepSpectralFun end

h_val(xs::Vector, ::SquareMMF) = sum(abs2, xs)
h_conj_dom(xs::Vector, ::SquareMMF) = true
h_conj(xs::Vector, ::SquareMMF) = sum((x >= 0 ? zero(x) : abs2(x / 2)) for x in xs)

h_der1(ds::Vector, xs::Vector, ::SquareMMF) = (@. ds = xs + xs)
h_der2(ds::Vector, xs::Vector, ::SquareMMF) = (@. ds = 2)
h_der3(ds::Vector, xs::Vector, ::SquareMMF) = (@. ds = 0)

function get_initial_point(d::Int, ::SquareMMF)
    # @warn("not initial central point") # TODO
    return (2d, 1, 1)
end


# power in (1,2]
# NOTE power 1 is homogeneous and just equal to trace of PSD matrix (a linear function)
# NOTE for power 2, use SquareMMF
struct Power12MMF <: SepSpectralFun
    p::Real
    Power12MMF(p::Real) = (@assert 1 < p <= 2; new(p))
end

h_val(xs::Vector, h::Power12MMF) = sum(x^h.p for x in xs)
h_conj_dom(xs::Vector, h::Power12MMF) = true
function h_conj(xs::Vector, h::Power12MMF)
    p = h.p
    q = p / (p - 1)
    return (p - 1) * sum((x >= 0 ? zero(x) : (abs(x) / p)^q) for x in xs)
end

function h_der1(ds::Vector, xs::Vector, h::Power12MMF)
    p = h.p
    pm1 = p - 1
    @. ds = p * xs^pm1
    return ds
end
function h_der2(ds::Vector, xs::Vector, h::Power12MMF)
    p = h.p
    pm2 = p - 2
    coef = p * (p - 1)
    @. ds = coef * xs^pm2
    return ds
end
function h_der3(ds::Vector, xs::Vector, h::Power12MMF)
    p = h.p
    pm3 = p - 3
    coef = p * (p - 1) * (p - 2)
    @. ds = coef * xs^pm3
    return ds
end

function get_initial_point(d::Int, h::Power12MMF)
    # @warn("not initial central point") # TODO
    return (2d, 1, 1)
end
