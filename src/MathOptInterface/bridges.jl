#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
bridge for MOI cones that are Hypatia cones up to a permutation of the
vectorization: MOI spectral and nuclear norm cones (transposing a matrix with
more rows than columns), and the scaled MOI Hermitian PSD cone (interleaving
real and imaginary parts), which MOI's SetDotScalingBridge reaches from the
MOI Hermitian PSD cone
=#

const PermutedCone = Union{
    MOI.NormSpectralCone,
    MOI.NormNuclearCone,
    MOI.Scaled{MOI.HermitianPositiveSemidefiniteConeTriangle},
}

struct PermuteBridge{T <: Real, S <: PermutedCone, S2} <:
       MOI.Bridges.Constraint.SetMapBridge{T, S2, S, VAF{T}, VAF{T}}
    constraint::MOI.ConstraintIndex{VAF{T}, S2}
    set::S
end

# below 1 so that the Hermitian PSD cone takes SetDotScalingBridge and this
# bridge (cost 1.1) rather than HermitianToSymmetricPSDBridge to a real PSD cone
MOI.Bridges.bridging_cost(::Type{<:PermuteBridge{<:Real, <:MOI.Scaled}}) = 0.1

function MOI.supports_constraint(
    ::Type{<:PermuteBridge{T}},
    ::Type{VAF{T}},
    ::Type{<:PermutedCone},
) where {T <: Real}
    return true
end

function MOI.Bridges.Constraint.concrete_bridge_type(
    ::Type{<:PermuteBridge{T}},
    ::Type{VAF{T}},
    ::Type{S},
) where {T <: Real, S <: PermutedCone}
    S2 = (S <: MOI.Scaled ? PosSemidefTriCone{T, Complex{T}} : EpiNormSpectralCone{T, T})
    return PermuteBridge{T, S, S2}
end

function MOI.Bridges.map_set(
    ::Type{<:PermuteBridge{T}},
    set::Union{MOI.NormSpectralCone, MOI.NormNuclearCone},
) where {T <: Real}
    dims = extrema((set.row_dim, set.column_dim))
    return EpiNormSpectralCone{T, T}(dims..., set isa MOI.NormNuclearCone)
end

function MOI.Bridges.map_set(
    ::Type{<:PermuteBridge{T}},
    set::MOI.Scaled{MOI.HermitianPositiveSemidefiniteConeTriangle},
) where {T <: Real}
    return PosSemidefTriCone{T, Complex{T}}(MOI.dimension(set))
end

MOI.Bridges.inverse_map_set(bridge::PermuteBridge, ::MOI.AbstractVectorSet) = bridge.set

function MOI.Bridges.Constraint.bridge_constraint(
    ::Type{PermuteBridge{T, S, S2}},
    model::MOI.ModelLike,
    func::VAF{T},
    set::S,
) where {T, S, S2}
    new_func = MOI.Utilities.eachscalar(func)[permute_idxs(set)]
    new_set = MOI.Bridges.map_set(PermuteBridge{T}, set)
    ci = MOI.add_constraint(model, new_func, new_set)
    return PermuteBridge{T, S, S2}(ci, set)
end

# the permutation depends on the set, not only on the bridge type
function MOI.Bridges.map_function(bridge::PermuteBridge, func)
    return MOI.Utilities.eachscalar(func)[permute_idxs(bridge.set)]
end

function MOI.Bridges.inverse_map_function(bridge::PermuteBridge, func)
    return MOI.Utilities.eachscalar(func)[invperm(permute_idxs(bridge.set))]
end

# a permutation is orthogonal
function MOI.Bridges.adjoint_map_function(bridge::PermuteBridge, func)
    return MOI.Bridges.inverse_map_function(bridge, func)
end

function MOI.Bridges.inverse_adjoint_map_function(bridge::PermuteBridge, func)
    return MOI.Bridges.map_function(bridge, func)
end

# SetMapBridge maps these with the bridge type, which lacks the set
function MOI.modify(
    model::MOI.ModelLike,
    bridge::PermuteBridge,
    chg::MOI.VectorConstantChange,
)
    new_constant = MOI.Bridges.map_function(bridge, chg.new_constant)
    MOI.modify(model, bridge.constraint, MOI.VectorConstantChange(new_constant))
    return
end

function MOI.modify(
    model::MOI.ModelLike,
    bridge::PermuteBridge{T},
    chg::MOI.MultirowChange{T},
) where {T}
    inv_perm = invperm(permute_idxs(bridge.set))
    new_coefs = [(inv_perm[i], c) for (i, c) in chg.new_coefficients]
    MOI.modify(model, bridge.constraint, MOI.MultirowChange(chg.variable, new_coefs))
    return
end

# SetDotScalingBridge drops zero coefficients from a MultirowChange before this
# bridge sees it, so refuse it and let a CachingOptimizer rebuild the model
function MOI.modify(
    ::MOI.ModelLike,
    ::PermuteBridge{T, MOI.Scaled{MOI.HermitianPositiveSemidefiniteConeTriangle}},
    chg::MOI.MultirowChange{T},
) where {T}
    return throw(MOI.Bridges.ModifyBridgeNotAllowed(chg))
end

function MOI.set(
    model::MOI.ModelLike,
    attr::MOI.ConstraintFunction,
    bridge::PermuteBridge{T},
    func::VAF{T},
) where {T}
    MOI.set(model, attr, bridge.constraint, MOI.Bridges.map_function(bridge, func))
    return
end

# setting ConstraintSet maps the set with the bridge, and a new shape would also
# change the permutation of the function
function MOI.Bridges.map_set(::PermuteBridge, ::PermutedCone)
    return throw(MOI.SetAttributeNotAllowed(MOI.ConstraintSet()))
end

function permute_idxs(set::Union{MOI.NormSpectralCone, MOI.NormNuclearCone})
    W_idxs = reshape(2:MOI.dimension(set), set.row_dim, set.column_dim)
    return vcat(1, vec(set.row_dim > set.column_dim ? W_idxs' : W_idxs))
end

function permute_idxs(set::MOI.Scaled{MOI.HermitianPositiveSemidefiniteConeTriangle})
    side = MOI.side_dimension(set)
    k_re = 1
    k_im = Cones.svec_length(side) + 1
    l = 1
    idxs = zeros(Int, MOI.dimension(set))
    for i in 1:side
        for j in 1:(i - 1)
            idxs[l] = k_re
            idxs[l + 1] = k_im
            k_re += 1
            k_im += 1
            l += 2
        end
        idxs[l] = k_re
        k_re += 1
        l += 1
    end
    return idxs
end
