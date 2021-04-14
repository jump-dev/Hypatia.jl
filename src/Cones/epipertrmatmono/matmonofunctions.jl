#=
TODO

suitable univariate matrix monotone functions and associated oracles
=#

struct NegLogMMF <: MatMonoFunction end

h_val(::Type{NegLogMMF}, x::Number) = -log(x)
h_sum(::Type{NegLogMMF}, xs::Vector{<:Number}) = -sum(log, xs)
h_der1(::Type{NegLogMMF}, x::Number) = -inv(x)
h_der2(::Type{NegLogMMF}, x::Number) = x^-2
h_der3(::Type{NegLogMMF}, x::Number) = -x^-3

function get_initial_point(::Type{NegLogMMF}, d::Int)
    # @warn("not initial central point") # TODO
    return (1, 1, 1)
end


# struct EntropyMMF <: MatMonoFunction end
#
# struct Power12MMF <: MatMonoFunction end
