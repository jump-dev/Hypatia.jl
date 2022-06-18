# Examples

See the Hypatia [examples folder](https://github.com/chriscoey/Hypatia.jl/tree/master/examples) for example models and corresponding test/benchmark instances.
Most examples have options for instance sizes/characteristics and formulation variants.
See the example files for more information and references.
Most example models use JuMP, and some use Hypatia's native interface.
New examples are welcomed and should be implemented similarly to the existing examples and linked from this page.

  - [CBLIB.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/CBLIB) Run a given CBLIB instance from the [Conic Benchmark Library](http://cblib.zib.de/).
  - [Central polynomial matrix.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/centralpolymat) Minimize a spectral function of a gram matrix of a polynomial.
  - [Classical-quantum capacity.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/classicalquantum) Compute the capacity of a classical-to-quantum channel.
  - [Condition number.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/conditionnum) Minimize the condition number of a matrix pencil subject to a linear matrix inequality.
  - [Contraction analysis.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/contraction) Find a contraction metric that guarantees global stability of a dynamical system.
  - [Convexity parameter.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/convexityparameter) Find the strong convexity parameter of a polynomial function over a domain.
  - [Covariance estimation.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/covarianceest) Estimate a covariance matrix that satisfies some given prior information and minimizes a given convex spectral function.
  - [Density estimation.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/densityest) Find a valid polynomial density function maximizing the likelihood of a set of observations.
  - [Discrete maximum likelihood.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/discretemaxlikelihood) Maximize the likelihood of some observations at discrete points, subject to the probability vector being close to a uniform prior.
  - [D-optimal design.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/doptimaldesign) Solve a D-optimal experiment design problem, i.e. maximize the determinant of the information matrix subject to side constraints.
  - [Entanglement-assisted capacity.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/entanglementassisted) Compute the entanglement-assisted classical capacity of a quantum channel.
  - [Experiment design.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/experimentdesign) Solve a general experiment design problem that minimizes a given convex spectral function of the information matrix subject to side constraints.
  - [Linear program.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/linearopt) Solve a simple linear program.
  - [Lotka-Volterra.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/lotkavolterra) Find an optimal controller for a Lotka-Volterra model of population dynamics.
  - [Lyapunov stability.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/lyapunovstability) Minimize an upper bound on the root mean square gain of a dynamical system.
  - [Matrix completion.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/matrixcompletion) Complete a rectangular matrix by minimizing the nuclear norm and constraining the missing entries.
  - [Matrix quadratic.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/matrixquadratic) Find a rectangular matrix that minimizes a linear function and satisfies a constraint on the outer product of the matrix.
  - [Matrix regression.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/matrixregression) Solve a multiple-output (or matrix) regression problem with regularization terms (such as ``\ell_1``, ``\ell_2``, or nuclear norm).
  - [Maximum volume hypercube.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/maxvolume) Find a maximum volume hypercube (with edges parallel to the axes) inside a given polyhedron or ellipsoid.
  - [Nearest correlation matrix.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/nearestcorrelation) Compute the nearest correlation matrix in the quantum relative entropy sense.
  - [Nearest polynomial matrix.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/nearestpolymat) Given a symmetric matrix of polynomials ``H``, find a polynomial matrix ``Q`` that minimizes the sum of the integrals of its elements over the unit box and guarantees ``Q - H`` is pointwise PSD on the unit box.
  - [Nearest PSD matrix.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/nearestpsd) Find a sparse PSD matrix or a PSD-completable matrix (with a given sparsity pattern) with constant trace that maximizes a linear function.
  - [Nonparametric distribution.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/nonparametricdistr) Given a random variable taking values in a finite set, compute the distribution minimizing a given convex spectral function over all distributions satisfying some prior information.
  - [Norm cone polynomial.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/normconepoly) Given a vector of polynomials, check a sufficient condition for pointwise membership in the epigraph of the ``\ell_1`` or ``\ell_2`` norm.
  - [Polynomial envelope.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/polyenvelope) Find a polynomial that closely approximates, over the unit box, the lower envelope of a given list of polynomials.
  - [Polynomial minimization.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/polymin) Compute a lower bound for a given polynomial over a given semialgebraic set.
  - [Polynomial norm.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/polynorm) Find a polynomial that, over the unit box, has minimal integral and belongs pointwise to the epigraph of the ``\ell_1`` or ``\ell_2`` norm of other given polynomials.
  - [Portfolio.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/portfolio) Maximize the expected returns of a stock portfolio and satisfy various risk constraints.
  - [Region of attraction.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/regionofattr) Find the region of attraction of a polynomial control system.
  - [Relative entropy of entanglement.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/relentrentanglement) Compute a lower bound on relative entropy of entanglement with a positive partial transpose relaxation.
  - [Robust geometric programming.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/robustgeomprog) Bound the worst-case optimal value of an uncertain signomial function with a given coefficient uncertainty set.
  - [Semidefinite polynomial matrix.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/semidefinitepoly) Check a sufficient condition for global convexity of a given polynomial.
  - [Shape constrained regression.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/shapeconregr) Given a dataset, fit a polynomial function that satisfies shape constraints such as monotonicity or convexity over a domain.
  - [Signomial minimization.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/signomialmin) Compute a global lower bound for a given signomial function.
  - [Sparse LMI.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/sparselmi) Optimize over a simple linear matrix inequality with sparse data.
  - [Sparse principal components.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/sparsepca) Solve a convex relaxation of the problem of approximating a symmetric matrix by a rank-one matrix with a cardinality-constrained eigenvector.
  - [Stability number.](https://github.com/chriscoey/Hypatia.jl/tree/master/examples/stabilitynumber) Given a graph, solve for a particular strengthening of the theta function towards the stability number.
