using Hypatia, DynamicPolynomials, Combinatorics

@polyvar x
@polyvar y

n = 2
hess_basis = [1, x, y]
poly_basis = [x ^ 2, y ^ 2, x * y, x ^ 2 * y, x * y ^ 2, x ^ 3, y ^ 3]
vars = [x, y]

# honestly dunno how to best type mono
function double_integrate(mono::Union{Monomial, Term, PolyVar}, wrt)
    ret = coefficient(mono) * Monomial(1)
    for (der, ntimes) in wrt
        if der in variables(mono) && degree(mono, der) > 0
            deg = degree(mono, der)
            if ntimes == 2
                ret *= inv((deg + 1) * (deg + 2)) * der ^ (deg + 2)
            elseif ntimes == 1
                ret *= inv(deg + 1) * der ^ (deg + 1)
            else
                ret *= der
            end
        else
            if ntimes == 2
                ret *= 0.5 * der ^ 2
            elseif ntimes == 1
                ret *= der
            end
        end
    end
    return ret
end

function double_integrate(poly, wrt)
    return sum(double_integrate(t, wrt) for t in terms(poly))
end

@assert double_integrate(x, Dict(x => 1, y => 1)) == 0.5 * x ^ 2 * y
@assert double_integrate(Monomial(1), Dict(x => 1, y => 1)) == x * y
@assert double_integrate(2 * x, Dict(x => 2, y => 0)) == 1 / 3 * x ^ 3
@assert double_integrate(2 * y * x^0, Dict(x => 0, y => 2)) == 1 / 3 * y ^ 3
@assert double_integrate(3 * Monomial(1), Dict(x => 0, y => 2)) == 1.5 * y ^ 2
@assert double_integrate(x, Dict(x => 0, y => 2)) == 0.5 * x * y ^ 2
double_integrate(2 * x + y + 3, Dict(x => 0, y => 2))

function special_integrator(n, hess_basis, poly_basis)
    U_hess = length(hess_basis)
    U_poly = length(poly_basis)
    dim = div(n * (n + 1), 2) * U_hess
    integrator = zeros(dim, dim)
    integrator_all = zeros(dim, dim)
    wrt = Dict(var => 0 for var in vars)
    idxs_affecting = [Dict() for u in 1:U_poly]
    coeffs_affecting = [Int[] for u in 1:U_poly]
    col = 1
    idx = 1
    for i in 1:n
        var1 = vars[i]
        wrt[var1] = 1
        for j in 1:i
            var2 = vars[j]
            if i == j
                wrt[var1] = 2
            else
                wrt[var2] = 1
            end
            for u in hess_basis
                integral = double_integrate(u, wrt)
                # @show u, integral
                for row in 1:U_poly
                    if poly_basis[row] in [t.x for t in terms(integral)]
                        if haskey(idxs_affecting[row], idx)
                            push!(idxs_affecting[row][idx], col)
                        else
                            idxs_affecting[row][idx] = [col]
                        end
                        # the only idx affecting is the one we are on currently
                        if length(keys(idxs_affecting[row])) == 1
                            integrator[row, col] = coefficient(integral, poly_basis[row])
                        end
                        integrator_all[row, col] = coefficient(integral, poly_basis[row])
                    end
                end
                col += 1
            end
            idx += 1
            wrt[var2] = 0
        end
        wrt[var1] = 0
    end
    additional_rows = 1
    @show idxs_affecting

    for u in 1:U_poly
        if length(keys(idxs_affecting[u])) > 1
            first_idx = first(first(idxs_affecting[u]))
            first_cols = last(first(idxs_affecting[u]))
            for (idx, cols) in idxs_affecting[u]
                if idx == first(first(idxs_affecting))
                    continue
                else
                    integrator[U_poly + additional_rows, first_cols] = -integrator_all[u, first_cols]
                    integrator[U_poly + additional_rows, cols] = integrator_all[u, cols]
                end
            end
            additional_rows += 1
        end
    end
    return integrator
end

integrator = special_integrator(n, hess_basis, poly_basis)

n = 2
hess_basis = [1, x + y + 2, x + 3]
poly_basis = [x ^ 2, y ^ 2, x * y, x ^ 2 * y, x * y ^ 2, x ^ 3, y ^ 3]
integrator = special_integrator(n, hess_basis, poly_basis)


convex_poly = x ^ 4 + y ^ 2
