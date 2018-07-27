

function clean_zeros!(v; tol=1e-10)
    for idx in eachindex(v)
        if abs(v[idx]) < tol
            v[idx] = 0.0
        end
    end
    return v
end

function cheb2_data(d::Int)
    @assert d > 1
    L = 1 + d
    U = 1 + 2d

    # U Chebyshev points of the second kind
    pts = [sinpi(j/(2(U-1))) for j in -(U-1):2:(U-1)]
    clean_zeros!(pts)

    # weights for Clenshaw-Curtis quadrature at pts
    wa = Float64[2/(1 - j^2) for j in 0:2:U-1]
    append!(wa, wa[floor(Int, U/2):-1:2])
    w = real.(FFTW.ifft(wa))
    w[1] = w[1]/2.0
    push!(w, w[1])
    clean_zeros!(w)

    # evaluations of Chebyshev polynomials of the first kind up to degree d at pts
    P0 = Matrix{Float64}(undef, U, L)
    P0[:,1] .= 1.0
    P0[:,2] .= pts
    for t in 3:L
        P0[:,t] .= 2.0.*pts.*P0[:,t-1] .- P0[:,t-2]
    end
    clean_zeros!(P0)

    # evaluations of a basis for the space of degree d polynomials at pts
    P = Array(qr(P0).Q)
    clean_zeros!(P)

    return (L=L, U=U, pts=pts, w=w, P0=P0, P=P)
end

function padua_data(d::Int)
    @assert d > 1
    n = 2
    L = binomial(n+d, n)
    U = binomial(n+2d, n)

    # Padua points for degree 2d
    cheba = [cospi(j/(2d)) for j in 0:2d]
    chebb = [cospi(j/(2d+1)) for j in 0:2d+1]
    pts = Matrix{Float64}(undef, U, n)
    j = 1
    for a in 0:2d
        for b in 0:2d+1
            if iseven(a+b)
                pts[j,1] = cheba[a+1]
                pts[U+1-j,2] = chebb[2d+2-b]
                j += 1
            end
        end
    end

    # evaluations of bivariate product Chebyshev polynomials of the first kind up to degree d at pts
    ua = Matrix{Float64}(undef, U, d+1)
    ua[:,1] .= 1.0
    ua[:,2] .= pts[:,1]
    for t in 3:d+1
        ua[:,t] .= 2.0.*ua[:,2].*ua[:,t-1] .- ua[:,t-2]
    end
    clean_zeros!(ua)
    ub = Matrix{Float64}(undef, U, d+1)
    ub[:,1] .= 1.0
    ub[:,2] .= pts[:,2]
    for t in 3:d+1
        ub[:,t] .= 2.0.*ub[:,2].*ub[:,t-1] .- ub[:,t-2]
    end
    clean_zeros!(ub)

    P0 = Matrix{Float64}(undef, U, L)
    P0[:,1] = ones(U)
    col = 1
    for t in 2:d+1
        col += 1
        P0[:,col] .= ua[:,t]
        col += 1
        P0[:,col] .= ub[:,t]
        if t > d
            break
        end
        for ad in Combinatorics.partitions(t, 2)
            col += 1
            P0[:,col] .= ua[:,ad[1]+1] .* ub[:,ad[2]+1]
            if ad[1] != ad[2]
                col += 1
                P0[:,col] .= ua[:,ad[2]+1] .* ub[:,ad[1]+1]
            end
        end
    end
    clean_zeros!(P0)

    # evaluations of a basis for the space of bivariate degree d polynomials at pts
    P = Array(qr(P0).Q)
    clean_zeros!(P)

    return (L=L, U=U, pts=pts, w=NaN, P0=P0, P=P)
end


# weights for cubature at pts in Padua
# TODO adapt matlab code
#   argn = linspace(0,pi,n+1);
#   argn1 = linspace(0,pi,n+2);
#   k = [0:2:n]';
#   l = (n-mod(n,2))/2+1;
# % even-degree Chebyshev polynomials on the subgrids
#   TE1 = cos(k*argn(1:2:n+1));
#   TE1(2:l,:) = TE1(2:l,:)*sqrt(2);
#   TO1 = cos(k*argn(2:2:n+1));
#   TO1(2:l,:) = TO1(2:l,:)*sqrt(2);
#   TE2 = cos(k*argn1(1:2:n+2));
#   TE2(2:l,:) = TE2(2:l,:)*sqrt(2);
#   TO2 = cos(k*argn1(2:2:n+2));
#   TO2(2:l,:) = TO2(2:l,:)*sqrt(2);
# % even,even moments matrix
#   mom = 2*sqrt(2)./(1-k.^2);
#   mom(1) = 2;
#   [M1,M2] = meshgrid(mom);
#   M = M1.*M2;
#   Mmom = fliplr(triu(fliplr(M)));
# % interpolation weights matrices
#   W1 = 2*ones(l)/(n*(n+1));
#   W2 = 2*ones((n+mod(n,2))/2+1,(n+mod(n,2))/2)/(n*(n+1));
#   W1(:,1) = W1(:,1)/2;
#   W2(1,:) = W2(1,:)/2;
#   if (mod(n,2) == 0)
#     Mmom(n/2+1,1) = Mmom(n/2+1,1)/2;
#     W1(:,n/2+1) = W1(:,n/2+1)/2;
#     W1(n/2+1,:) = W1(n/2+1,:)/2;
#   else
#     W2((n+1)/2+1,:) = W2((n+1)/2+1,:)/2;
#     W2(:,(n+1)/2) = W2(:,(n+1)/2)/2;
#   end
# % cubature weights as matrices on the subgrids.
#   L1 = W1.*(TE1'*Mmom*TO2)';
#   L2 = W2.*(TO1'*Mmom*TE2)';
#   if (mod(n,2) == 0)
#     L = zeros(n/2+1,n+1);
#     L(:,1:2:n+1) = L1;
#     L(:,2:2:n+1) = L2;
#     L = L(:);
#   else
#     L = zeros((n+1)/2,(n+2));
#     L = [L1',L2']';
#     L = L(:);
#   end


# TODO n >= 3
# function approx_fekete_data(n::Int, d::Int)
