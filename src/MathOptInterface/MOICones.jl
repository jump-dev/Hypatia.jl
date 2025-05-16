module MOICones

import MathOptInterface
const MOI = MathOptInterface

abstract type MOICone <: MOI.AbstractVectorSet end

function dimension(cone::MOICone)
    return error("dimension($cone) not implemented")
end

MOI.dimension(cone::MOICone) = dimension(cone)

function cone_from_moi(::Type{<:Real}, cone::MOICone)
    return error("cone_from_moi($cone) not implemented")
end

end

function cone_from_moi(t::Type{<:Real}, cone::MOICones.MOICone)
    return MOICones.cone_from_moi(t, cone)
end
