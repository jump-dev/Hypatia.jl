# Conic models

## Conic form

Any convex optimization problem may be represented as a conic problem that minimizes a linear function over the intersection of an affine subspace with a Cartesian product of primitive proper cones (i.e. irreducible, closed, convex, pointed, and full-dimensional conic sets).
An advantage of using conic form is that a conic problem, if well-posed, has a very simple and easily checkable certificate of optimality, primal infeasibility, or dual infeasibility.

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

## Model interface

```@meta
CurrentModule = Hypatia.Models
```

Hypatia's [`Models`](@ref) module specifies a [`Model`](@ref) type corresponding to the primal conic form above:

  - ``\mathcal{K}`` is a vector of Hypatia cones,
  - ``c \in \mathbb{R}^n``, ``b \in \mathbb{R}^p``, ``h \in \mathbb{R}^q`` are vectors,
  - ``A \in \mathbb{R}^{p \times n}`` and ``G \in \mathbb{R}^{q \times n}`` are linear operators.

An objective offset can be specified using the keyword arg `obj_offset` (the default is 0).
See [Models module](@ref).

## Polynomial utilities

```@meta
CurrentModule = Hypatia.PolyUtils
```

The [`PolyUtils`](@ref) module provides tools for setting up polynomial interpolations for sum-of-squares models; see [PolyUtils module](@ref).
