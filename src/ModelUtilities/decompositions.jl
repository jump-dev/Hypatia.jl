#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import JuMP
import DynamicPolynomials
import Combinatorics

function get_decomposition(
        primal_var,
        constraint,
        lambda_oracle,
        hessian_oracle,
        n,
        d,
        )
    s = JuMP.value(primal_var)
    x = JuMP.dual(constraint)
    w = hessian_oracle(x) \ s
    lambda_inv = inv(lambda(x))
    S = lambda_inv * lambda_oracle(w) * lambda_inv
    f = cholesky(f)
    basis = build_basis(n, d)
    decomposition = f.U * basis
    return decomposition
end

function calc_u(d, monovec)
    n = length(monovec)
    u = Vector{Vector}(undef, n) # TODO type properly
    for j in 1:n
        uj = u[j] = Vector(undef, d+1) # TODO type properly
        uj[1] = Monomial(1)
        uj[2] = monovec[j]
        for t in 3:(d + 1)
            uj[t] = 2.0 * uj[2] * uj[t-1] - uj[t-2]
        end
    end
    return u
end

# returns the basis dynamic polynomials
function build_basis(n, d)
    @polyvar x[1:n]
    u = calc_u(d, x)
    L = binomial(n + d, n)
    m = Vector{Float64}(undef, L)
    m[1] = 2^n
    M = Vector(undef, L)
    M[1] = Monomial(1)

    col = 1
    for t in 1:d
        for xp in Combinatorics.multiexponents(n, t)
            col += 1
            if any(isodd, xp)
                m[col] = 0.0
            else
                m[col] = m[1] / prod(1.0 - abs2(xp[j]) for j in 1:n)
            end
            M[col] = u[1][xp[1] + 1]
            for j in 2:n
                M[col] *= u[j][xp[j] + 1]
            end
        end
    end
    return M
end
