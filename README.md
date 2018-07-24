# Alfonso.jl
(under construction) unofficial Julia re-implementation of alfonso (ALgorithm FOr Non-Symmetric Optimization), originally by D. Papp and S. Yıldız: https://github.com/dpapp-github/alfonso

## TODO
- construct barriers internally from recognized cones and set up efficient oracles
- use new Julia logging tools instead of print https://docs.julialang.org/en/latest/stdlib/Logging/
- allow generic number types and generic matrices/vectors (eg sparse or dense) - should these be determined by inputs, or by options?
- improve use of sparse linear algebra, especially in linear system solve (lhs matrix has special structure: mostly symmetric psd plus a skew-symmetric row/column from b)
- hook up to MathOptInterface
- move to JuliaOpt
- try supporting quadratic objectives (as described by [Vandenberghe](http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf))
