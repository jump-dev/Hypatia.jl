module MOICones

import Hypatia.Cones: Cone
import MathOptInterface
const MOI = MathOptInterface

# This stopgap interface is used by the predefined cones in src/MathOptInterface/cones.jl
# until they are migrated to the MOIWrapper interface.
abstract type MOICone <: MOI.AbstractVectorSet end

function dimension(cone::MOICone)
    return error("dimension($cone) not implemented")
end

MOI.dimension(cone::MOICone) = dimension(cone)

function cone_from_moi(::Type{<:Real}, cone::MOICone)
    return error("cone_from_moi($cone) not implemented")
end

# "Wrapper" for Hypatia cones to be used with MathOptInterface
struct MOIWrapper{T <:Real} <: MOICone
    cone::Cone{T}

    function MOIWrapper{T}(cone::Cone{T}) where {T <: Real}
        return new{T}(cone)
    end
end

MOI.dimension(cone::MOIWrapper) = cone.cone.dim

end

# "Re-export" functions for use in src/MathOptInterface/wrapper.jl
function cone_from_moi(t::Type{<:Real}, cone::MOICones.MOICone)
    return MOICones.cone_from_moi(t, cone)
end

# "Re-export" functions for use in src/MathOptInterface/wrapper.jl
function cone_from_moi(t::Type{<:Real}, cone::MOICones.MOIWrapper)
    return cone.cone
end
