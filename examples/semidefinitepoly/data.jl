#=
list of predefined matrix polynomials
=#

import DynamicPolynomials
const DP = DynamicPolynomials

function get_psdpoly_data(matpoly::Symbol)
    if matpoly == :matpoly1
        DP.@polyvar x
        M = [
            (x + 2x^3)  1;
            (-x^2 + 2)  (3x^2 - x + 1);
            ]
        MM = M' * M
        return ([x], MM, true)
    elseif matpoly == :matpoly2
        DP.@polyvar x
        poly = x^4 + 2x^2
        return ([x], poly, true)
    elseif matpoly == :matpoly3
        DP.@polyvar x y
        poly = (x + y)^4 + (x + y)^2
        return ([x, y], poly, true)
    elseif matpoly == :matpoly4
        n = 3
        m = 3
        d = 1
        DP.@polyvar x[1:n]
        Z = DP.monomials(x, 0:d)
        M = [sum(rand() * Z[l] for l in 1:length(Z)) for i in 1:m, j in 1:m]
        MM = M' * M
        MM = 0.5 * (MM + MM')
        return (x, MM, true)
    elseif matpoly == :matpoly5
        # example modified from
        # https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/simplematrixsos.jl
        # Example 3.77 and 3.79 of Blekherman, G., Parrilo, P. A., & Thomas, R. R. (Eds.),
        # Semidefinite optimization and convex algebraic geometry SIAM 2013
        DP.@polyvar x
        P = [
            (x^2 - 2x + 2)  x;
            x               x^2;
            ]
        return ([x], P, true)
    elseif matpoly == :matpoly6
        # example modified from
        # https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/choi.jl
        # verifies that a given polynomial matrix is not a Sum-of-Squares matrix
        # see Choi, M. D., "Positive semidefinite biquadratic forms",
        # Linear Algebra and its Applications, 1975, 12(2), 95-100
        DP.@polyvar x y z
        P = [
            (x^2 + 2y^2)    (-x * y)        (-x * z);
            (-x * y)        (y^2 + 2z^2)    (-y * z);
            (-x * z)        (-y * z)        (z^2 + 2x^2);
            ] .* (x * y * z)^0
        # TODO the (x * y * z)^0 can be removed when
        # https://github.com/JuliaOpt/SumOfSquares.jl/issues/106 is fixed
        return ([x, y, z], P, false)
    elseif matpoly == :matpoly7
        # example modified from
        # https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/sosdemo9.jl
        # Section 3.9 of SOSTOOLS User's Manual, see
        # https://www.cds.caltech.edu/sostools/
        DP.@polyvar x y z
        P = hvcat((2, 2),
            (x^4 + x^2 * y^2 + x^2 * z^2),
            (x * y * z^2 - x^3 * y - x * y * (y^2 + 2 * z^2)),
            (x * y * z^2 - x^3 * y - x * y * (y^2 + 2 * z^2)),
            (x^2 * y^2 + y^2 * z^2 + (y^2 + 2 * z^2)^2))
        return ([x, y, z], P, true)
    end
end
