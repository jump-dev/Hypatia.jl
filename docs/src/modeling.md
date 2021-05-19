## Cones, conic form, and certificates

```@contents
Pages = ["modeling.md"]
Depth = 4
```

Any convex optimization problem may be represented as a conic problem that minimizes a linear function over the intersection of an affine subspace with a Cartesian product of primitive proper cones (i.e. irreducible, closed, convex, pointed, and full-dimensional conic sets).
An advantage of using conic form is that a conic problem, if well-posed, has a very simple and easily checkable certificate of optimality, primal infeasibility, or dual infeasibility.

### Conic form

Hypatia's primal conic form over variable ``x \in \mathbb{R}^n`` is:
```math
\begin{aligned}
\min \quad c'x &:
\\
b - Ax &= 0
\\
h - Gx &\in \mathcal{K}
\end{aligned}
```
where ``\mathcal{K}`` is a proper cone.

The corresponding conic dual form over variables ``y \in \mathbb{R}^p`` and ``z \in \mathbb{R}^q`` is:
```math
\begin{aligned}
\max \quad -b'y - h'z &:
\\
c + A'y + G'z &= 0
\\
z &\in \mathcal{K}^*
\end{aligned}
```
where ``\mathcal{K}^*`` is the dual cone of ``\mathcal{K}``.

### Model interface

See [Models module](@ref) for Hypatia's model type.
A model specifies the data in the primal conic form above:
- ``\mathcal{K}`` is a vector of Hypatia cones,
- ``c \in \mathbb{R}^n``, ``b \in \mathbb{R}^p``, ``h \in \mathbb{R}^q`` are vectors,
-  ``A \in \mathbb{R}^{p \times n}`` and ``G \in \mathbb{R}^{q \times n}`` are linear operators.
An objective offset can be specified using the keyword arg `obj_offset` (the default is 0).

### Predefined cones

See [Cones module](@ref) for a list of Hypatia's predefined cone types.

### Generic cone interface

The cone interface allows specifying proper cones.
Hypatia predefines many proper cones that are practically useful; see the [Cones folder](https://github.com/chriscoey/Hypatia.jl/tree/master/src/Cones).
Defining a proper cone requires implementing several key oracles:
- an initial interior point inside the cone,
- a feasibility test, which checks whether a given point is in the interior of the cone,
- gradient and Hessian evaluations for a logarithmically homogeneous self-concordant barrier function for the cone.
Additional optional oracle can be specified to improve speed and numerical performance.
Defining a new cone automatically defines its dual cone (through the `use_dual` option) also.
