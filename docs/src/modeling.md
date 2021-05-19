## Cones, conic form, and certificates

```@contents
Pages = ["modeling.md"]
Depth = 4
```

A proper cone is a ...

### Conic form

The primal conic form over variable ``x \in \mathbb{R}^n`` is:
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

A model specifies the data in the primal conic form above:
- ``\mathcal{K}`` is a vector of Hypatia cones,
- ``c \in \mathbb{R}^n``, ``b \in \mathbb{R}^p``, ``h \in \mathbb{R}^q`` are vectors,
-  ``A \in \mathbb{R}^{p \times n}`` and ``G \in \mathbb{R}^{q \times n}`` are linear operators.
An objective offset can be specified using the keyword arg `obj_offset` (the default is 0).

### Predefined cones

list ...

### Generic cone interface

The cone interface allows specifying proper cones.
Hypatia predefines many proper cones that are practically useful; see the [Cones folder](https://github.com/chriscoey/Hypatia.jl/tree/master/src/Cones).
Defining a proper cone requires implementing several key oracles:
- an initial interior point inside the cone,
- a feasibility test, which checks whether a given point is in the interior of the cone,
- gradient and Hessian evaluations for a logarithmically homogeneous self-concordant barrier function for the cone.
Additional optional oracle can be specified to improve speed and numerical performance.
Defining a new cone automatically defines its dual cone (through the `use_dual` option) also.
