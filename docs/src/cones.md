# Cone interface and predefined cones

```@meta
CurrentModule = Hypatia.Cones
```

## Generic cone interface

Hypatia's [`Cones`](@ref) module specifies a generic cone interface that allows defining new proper cones as subtypes of [`Cone`](@ref).
This requires implementing cone oracles as methods for the new cone type; see [Cone oracles](@ref).
The required oracles are:
- an initial interior point inside the cone,
- a feasibility test, which checks whether a given point is in the interior of the cone,
- gradient and Hessian evaluations for a logarithmically homogeneous self-concordant barrier function for the cone.
Additional optional oracles can be specified to improve speed and numerical performance.
Defining a new cone automatically defines its dual cone (through the `use_dual` option) also.

## Predefined cones

Hypatia predefines many proper cones that are practically useful; see [Predefined cone types](@ref).
These cones are used in Hypatia's [Examples](@ref) and [native instances](https://github.com/chriscoey/Hypatia.jl/blob/master/test/nativeinstances.jl).
These cones are also wrapped as `MathOptInterface.AbstractVectorSet` types and exported from Hypatia; see [MathOptInterface cones](@ref).
