# Cone interface and predefined cones

```@meta
CurrentModule = Hypatia.Cones
```

## Generic cone interface

Hypatia's [`Cones`](@ref) module specifies a generic cone interface that allows defining new proper cones as subtypes of [`Cone`](@ref).
This requires implementing cone oracles as methods for the new cone type; see [Cone oracles](@ref).
The required oracles are:

  - an initial interior point inside the cone; see [`set_initial_point!`](@ref),
  - a feasibility test, which checks whether a given point is in the interior of the cone; see [`is_feas`](@ref),
  - gradient and Hessian evaluations for a logarithmically homogeneous self-concordant barrier (LHSCB) function for the cone; see [`grad`](@ref) and [`hess`](@ref).

Additional optional oracles can be specified to improve speed and numerical performance.
Defining a new cone automatically defines its dual cone (through the `use_dual` option) also.
See Hypatia's predefined cones in the [cones folder](https://github.com/chriscoey/Hypatia.jl/tree/master/src/Cones) for examples of how to implement a new cone type and efficient oracles.
The implementations of the [`HypoPowerMean`](@ref) cone (which uses a primal LHSCB) and the [`WSOSInterpNonnegative`](@ref) cone (which uses a dual LHSCB) are fairly typical.

## Predefined cones

Hypatia predefines many proper cones that are practically useful; see [Predefined cone types](@ref).
These cones are used in Hypatia's [Examples](@ref) and [native instances](https://github.com/chriscoey/Hypatia.jl/blob/master/test/nativeinstances.jl).
These cones are also wrapped as `MathOptInterface.AbstractVectorSet` types and exported from Hypatia; see [MathOptInterface cones](@ref).
