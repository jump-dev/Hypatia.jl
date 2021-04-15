#=
suitable univariate functions with matrix monotone derivative, and associated oracles

TODO add some tests for these
=#

h_sum(F::Type{<:SepSpectralFun}, xs::Vector{<:Number}) = sum(h_val(F, x_i) for x_i in xs)

# negative logarithm: -log(x)
struct NegLogMMF <: SepSpectralFun end

h_val(::Type{NegLogMMF}, x::Number) = -log(x)
h_conj(::Type{NegLogMMF}, x::Number) = -1 - log(x)
h_der1(::Type{NegLogMMF}, x::Number) = -inv(x)
h_der2(::Type{NegLogMMF}, x::Number) = x^-2
h_der3(::Type{NegLogMMF}, x::Number) = -2 * x^-3

function get_initial_point(::Type{NegLogMMF}, d::Int)
    # @warn("not initial central point") # TODO
    return (1, 1, 1)
end


# negative entropy: x * log(x)
struct NegEntropyMMF <: SepSpectralFun end

h_val(::Type{NegEntropyMMF}, x::Number) = x * log(x)
h_conj(::Type{NegEntropyMMF}, x::Number) = exp(1 + x)
h_der1(::Type{NegEntropyMMF}, x::Number) = 1 + log(x)
h_der2(::Type{NegEntropyMMF}, x::Number) = inv(x)
h_der3(::Type{NegEntropyMMF}, x::Number) = -x^-2

function get_initial_point(::Type{NegEntropyMMF}, d::Int)
    # @warn("not initial central point") # TODO
    return (1, 1, 1)
end


# # power in [1,2]
# # TODO or parametrize the type?
# struct Power12MMF <: SepSpectralFun
#     power::Real
#     function Power12MMF(power::Real)
#         @assert 1 <= power <= 2
#         return new(power)
#     end
# end
#
# h_val(f::Type{Power12MMF}, x::Number) = x^(f.power)
# h_conj(::Type{Power12MMF}, x::Number) =
# h_der1(::Type{Power12MMF}, x::Number) =
# h_der2(::Type{Power12MMF}, x::Number) =
# h_der3(::Type{Power12MMF}, x::Number) =
#
# function get_initial_point(::Type{Power12MMF}, d::Int)
#     # @warn("not initial central point") # TODO
#     return (1, 1, 1)
# end
