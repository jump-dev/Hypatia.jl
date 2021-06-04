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

## Example (for advanced users)

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

you don't have to contribute to Hypatia but to make things simpler in this example, let's assume we want to add `MyNewCone` to Hypatia.

### The cone struct

first two blocks are fields common to all cones, not all are required but this is an efficient way to set things up.
last four fields are specific to this cone.
notice that we use `A` to infer the dimension of the cone, which is stored in the required field `dim`.
we use the internal function `svec_side` to calculate the side dimension of ``smat(A w)``.

```jl
mutable struct MyNewCone{T <: Real} <: Cone{T}
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
    hess_fact_cache

    # fields specific to this cone
    A
    side
    fact
    rt2

    function MyNewCone{T}(
        A;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        # these are some standard fields
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.hess_fact_cache = hess_fact_cache

        # infer cone dimension from size of A
        cone.dim = size(A, 2)
        # cache the side dimesnion of smat(A * w)
        cone.side = svec_side(T, cone.dim)
        # cache √2 for convencience
        cone.rt2 = sqrt(T(2))

        return cone
    end
end
```

### Barrier parameter

in this example the barrier parameter for this cone is equal to the side dimension of ``smat(A w)``.
we already calculated this and stored it in `cone.side`.

```jl
get_nu(cone::MyNewCone) = cone.side
```

for an initial point we will use the `w` with the smallest norm such that
$$
mat(A w) = svec(I)
$$

### Initial interior point

```jl
function set_initial_point!(
    arr::AbstractVector{T},
    cone::MyNewCone{T},
    ) where {T <: Real}
    I_vec = smat_to_svec!(zeros(cone.dim), I(cone.side), cone.rt2)
    arr = A \ I_vec
    return arr
end
```

### Feasibility test

to test feasibility we calculate ``smat(A * x)`` and attempt a Cholesky factorization for the result.
if the factorization fails we reject the point, otherwise, we accept the point.
we will cache the factorization object in `cone.fact` for use in other oracles later (see for example update_grad).
the point is always stored in `cone.point`.

we will use the internal function [`svec_to_smat!`](@ref) for converting ``A * w`` into a matrix.
note that only the upper triangle is filled.

```jl
function update_feas(cone::MyNewCone{T}) where {T <: Real}
    side = cone.side
    w = cone.point
    temp = zeros(T, side, side)

    Aw_vec = A * w
    Aw_mat = svec_to_smat!(temp, Aw_vec, cone.rt2)
    fact = cholesky(Symmetric(Aw_mat, :U), check = false)
    cone.fact = fact  
    cone.is_feas = isposdef(Aw_fact)

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
function update_grad(cone::MyNewCone)
    Aw_inv = inv(cone.fact)
    smat_to_svec!(cone.grad, -Aw_inv, cone.rt2)

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

note that ``A \\delta`` when ``\\delta`` is the `k`th column of the identity is equal to the (scaled) `k`th column of ``A``.

```jl
function update_hess(cone::MyNewCone{T}) where {T <: Real}
    H = cone.hess.data
    A = cone.A
    rt2 = cone.rt2
    temp1 = zeros(T, side, side)
    temp2 = zeros(T, size(A, 1))

    k = 1
    for j in 1:cone.side, i in 1:j
        Aij = A[:, k]
        Aij_mat = svec_to_smat!(temp1, Aij, rt2)
        H[:, k] = A' * smat_to_svec!(temp2, Aw_inv * Aij_mat * Aw_inv, rt2)
        k += 1
    end

    cone.hess_updated = true
    return cone.hess
end
```

### Test the cone

see ... for examples for how to write a test

### (Optional) Making the cone available from MOI

see ... for examples for how to make your cone available
