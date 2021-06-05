# Cone interface and predefined cones

```@meta
CurrentModule = Hypatia.Cones
```

## Generic cone interface

Hypatia's [`Cones`](@ref) module specifies a generic cone interface that allows defining new proper cones as subtypes of [`Cone`](@ref).
This requires implementing cone oracles as methods for the new cone type; see [Cone oracles](@ref).
The required oracles are:
- an initial interior point inside the cone; see [`set_initial_point!`](@ref),
- the barrier parameter; see [`get_nu!`](@ref),
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

## Example (extra for experts)

this example is just one way to set up a new cone, there are other ways / it is possible to omit some fields.
the example is also meant to describe the interface, it's not a recommended implementation.
Suppose we want to define the spectrahedral cone:
$$
\{ w \\in R^d: smat(A w) \\succeq 0 \}
$$

a barrier for this cone is:
$$
-logdet(smat(A w))
$$

this cone is parametrized by data ``A``, so when we set it up we ask the user to provide it.

to add a cone you don't have to contribute to Hypatia but to make things simpler in this example, let's assume we want to add `MySpectrahedron` to Hypatia.

### The cone struct

first two blocks are fields common to all cones, not all are required but this is an efficient way to set things up.
last four fields are specific to this cone.
notice that we use `A` to infer the dimension of the cone, which is stored in the required field `dim`.
we use the internal function `svec_side` to calculate the side dimension of ``smat(A w)``.

```jl
mutable struct MySpectrahedron{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    # fields specific to this cone
    A
    side
    fact
    Aw_inv
    rt2

    function MySpectrahedron{T}(
        A;
        use_dual::Bool = false,
        ) where {T <: Real}
        cone = new{T}()
        cone.use_dual_barrier = use_dual

        cone.A = A
        (d1, d2) = size(A)
        # infer cone dimension from size of A
        cone.dim = d2
        # cache the side dimesnion of smat(A * w)
        cone.side = svec_side(T, d1)
        # cache √2 for convencience
        cone.rt2 = sqrt(T(2))

        return cone
    end
end
```

### Barrier parameter

in this example the barrier parameter for this cone is equal to the side dimension of ``smat(A w)``, which is `cone.side`.

```jl
get_nu(cone::MySpectrahedron) = cone.side
```

### Initial interior point

for an initial point we will use the `w` with the smallest norm such that
$$
A w = svec(I)
$$

```jl
function set_initial_point!(
    arr::AbstractVector{T},
    cone::MySpectrahedron{T},
    ) where {T <: Real}
    side = cone.side
    A = cone.A
    temp = zeros(T, size(A, 1))
    I_vec = smat_to_svec!(temp, Matrix{T}(I, side, side), cone.rt2)
    arr .= A \ I_vec
    return arr
end
```
the internal function [`smat_to_svec!`](@ref) is used to convert ``I`` into svec form.

### Feasibility test

to test feasibility we calculate ``smat(A * x)`` and attempt a Cholesky factorization for the result.
if the factorization fails we reject the point, otherwise, we accept the point.
we will cache the factorization object in `cone.fact` for use in other oracles later (see for example update_grad).
the point is always stored in `cone.point`.

we will use the internal function [`svec_to_smat!`](@ref) for converting ``A * w`` into a matrix.
note that only the upper triangle is filled.

```jl
function update_feas(cone::MySpectrahedron{T}) where {T <: Real}
    side = cone.side
    A = cone.A
    w = cone.point
    temp = zeros(T, side, side)

    Aw_vec = A * w
    Aw_mat = svec_to_smat!(temp, Aw_vec, cone.rt2)
    fact = cholesky(Symmetric(Aw_mat, :U), check = false)
    cone.fact = fact
    cone.is_feas = isposdef(fact)

    cone.feas_updated = true
    return cone.is_feas
end
```

### Gradient of the LHSCB

the gradient of the LHSCB is given by:
$$
\\nabla f (w) = A' svec((mat(A w))^{-1})
$$

we should be able to safely assume that whenever the gradient oracle is called, point in the cone was already checked in `update_feas`.
therefore we can reuse any data calculated in `update_feas`.
in particular, we have already calculated `mat(A * w)` and its factorization, so we will reuse it to compute `(mat(A * w))^{-1}`.

```jl
function update_grad(cone::MySpectrahedron{T}) where {T <: Real}
    A = cone.A
    Aw_inv = cone.Aw_inv = inv(cone.fact)
    temp = zeros(T, size(A, 1))
    smat_to_svec!(temp, -Aw_inv, cone.rt2)
    cone.grad = A' * temp

    cone.grad_updated = true
    return cone.grad
end
```

### Hessian of the LHSCB

the Hessian of the LHSCB is the operator such that:
$$
\\nabla^2 f (w) [\\delta] = A' svec((mat(A w))^{-1} mat(A \\delta) mat(A w))^{-1})
$$
for the explicit Hessian we apply ``\\nabla^2 f (w) `` to the columns of the (smat-ed) identity in ``R^{dim}``.

```jl
function update_hess(cone::MySpectrahedron{T}) where {T <: Real}
    isdefined(cone, :hess) || alloc_hess!(cone)
    side = cone.side
    A = cone.A
    Aw_inv = cone.Aw_inv
    rt2 = cone.rt2
    temp1 = zeros(T, side, side)
    temp2 = zeros(T, size(A, 1))
    H = cone.hess.data

    for k in 1:cone.dim
        Ak = A[:, k]
        Ak_mat = Symmetric(svec_to_smat!(temp1, Ak, rt2), :U)
        H[:, k] = A' * smat_to_svec!(temp2, Aw_inv * Ak_mat * Aw_inv, rt2)
    end

    cone.hess_updated = true
    return cone.hess
end
```

### TOO of the LHSCB

we won't implement a TOO.

```jl
use_dder3(::MySpectrahedron)::Bool = false
```

### Test the cone

see ... for examples for how to write a test

### (Optional) Making the cone available from MOI

see ... for examples for how to make your cone available
