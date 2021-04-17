#=
suitable univariate functions defined on ℝ₊₊ with matrix monotone derivative, and associated oracles
=#

# negative logarithm: -log(x)
struct NegLogMMF <: SepSpectralFun end

h_val(::Type{NegLogMMF}, xs::Vector) = -sum(log, xs)
h_conj_dom(::Type{NegLogMMF}, xs::Vector) = all(>(eps(eltype(xs))), xs)
h_conj(::Type{NegLogMMF}, xs::Vector) = -length(xs) - sum(log, xs) # TODO check

h_der1(::Type{NegLogMMF}, x::Real) = -inv(x)
h_der2(::Type{NegLogMMF}, x::Real) = x^-2
h_der3(::Type{NegLogMMF}, x::Real) = -2 * x^-3

function get_initial_point(::Type{NegLogMMF}, d::Int)
    # @warn("not initial central point") # TODO
    return (1, 1, 1)
end


# negative entropy: x * log(x)
struct NegEntropyMMF <: SepSpectralFun end

h_val(::Type{NegEntropyMMF}, xs::Vector) = sum(x * log(x) for x in xs)
h_conj_dom(::Type{NegEntropyMMF}, ::Vector) = true
h_conj(::Type{NegEntropyMMF}, xs::Vector) = sum(exp(-x - 1) for x in xs) # TODO check

h_der1(::Type{NegEntropyMMF}, x::Real) = 1 + log(x)
h_der2(::Type{NegEntropyMMF}, x::Real) = inv(x)
h_der3(::Type{NegEntropyMMF}, x::Real) = -x^-2

function get_initial_point(::Type{NegEntropyMMF}, d::Int)
    # @warn("not initial central point") # TODO
    return (1, 1, 1)
end


# square: x^2
struct SquareMMF <: SepSpectralFun end

h_val(::Type{SquareMMF}, xs::Vector) = sum(abs2, xs)
h_conj_dom(::Type{SquareMMF}, ::Vector) = true
h_conj(::Type{SquareMMF}, xs::Vector) = sum((x >= 0 ? zero(x) : abs2(x / 2)) for x in xs) # TODO check

h_der1(::Type{SquareMMF}, x::Real) = 2 * x
h_der2(::Type{SquareMMF}, x::Real) = 2
h_der3(::Type{SquareMMF}, x::Real) = 0 # TODO ?

function get_initial_point(::Type{SquareMMF}, d::Int)
    # @warn("not initial central point") # TODO
    return (2d, 1, 1)
end



# TODO power 1 is homogeneous so don't need persp, and is just equal to trace, so is a linear function anyway. so maybe exclude it.
# # power in (1,2]
# # TODO or parametrize the type?
# struct Power12MMF <: SepSpectralFun
#     power::Real
#     function Power12MMF(power::Real)
#         @assert 1 < power <= 2
#         return new(power)
#     end
# end
#
# h_val(f::Type{Power12MMF}, x::Real) = x^(f.power)
# h_conj(::Type{Power12MMF}, x::Real) =
# h_der1(::Type{Power12MMF}, x::Real) =
# h_der2(::Type{Power12MMF}, x::Real) =
# h_der3(::Type{Power12MMF}, x::Real) =
#
# function get_initial_point(::Type{Power12MMF}, d::Int)
#     # @warn("not initial central point") # TODO
#     return (1, 1, 1)
# end
