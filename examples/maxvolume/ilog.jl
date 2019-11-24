# modified from https://github.com/JuliaOpt/MathOptInterface.jl/blob/master/src/Bridges/Constraint/geomean.jl
function _ilog2(n, i)
    if n <= (one(n) << i)
        i
    else
        _ilog2(n, i+1)
    end
end
function ilog2(n::Integer)
    @assert n > zero(n)
    _ilog2(n, zero(n))
end