#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from files in https://github.com/dpapp-github/alfonso/
https://github.com/dpapp-github/alfonso/blob/master/ChebInterval.m
https://github.com/dpapp-github/alfonso/blob/master/PaduaSquare.m
https://github.com/dpapp-github/alfonso/blob/master/FeketeCube.m
and from Matlab files obtained from the packages
Chebfun http://www.chebfun.org/
Padua2DM by M. Caliari, S. De Marchi, A. Sommariva, and M. Vianello http://profs.sci.univr.it/~caliari/software.htm
=#


# return k Chebyshev points of the second kind
cheb2_pts(k::Int) = [-cospi(j/(k-1)) for j in 0:k-1]


function calc_u(n, d, pts)
    u = Vector{Matrix{Float64}}(undef, n)
    for j in 1:n
        uj = u[j] = Matrix{Float64}(undef, size(pts, 1), d+1)
        uj[:,1] .= 1
        uj[:,2] .= pts[:,j]
        for t in 3:d+1
            uj[:,t] .= 2 .* uj[:,2] .* uj[:,t-1] .- uj[:,t-2]
        end
    end
    return u
end


function cheb2_data(d::Int)
    @assert d > 1
    L = 1 + d
    U = 1 + 2d

    # Chebyshev points for degree 2d
    pts = cheb2_pts(U)

    # evaluations
    P0 = calc_u(1, d, pts)[1]
    P = Array(qr(P0).Q)

    # weights for Clenshaw-Curtis quadrature at pts
    wa = Float64[2/(1 - j^2) for j in 0:2:U-1]
    append!(wa, wa[floor(Int, U/2):-1:2])
    w = real.(FFTW.ifft(wa))
    w[1] = w[1]/2
    push!(w, w[1])

    return (L=L, U=U, pts=pts, P0=P0, P=P, w=w)
end


function padua_data(d::Int)
    @assert d > 1
    L = binomial(2+d, 2)
    U = binomial(2+2d, 2)

    # Padua points for degree 2d
    cheba = cheb2_pts(2d+1)
    chebb = cheb2_pts(2d+2)
    pts = Matrix{Float64}(undef, U, 2)
    j = 1
    for a in 0:2d
        for b in 0:2d+1
            if iseven(a+b)
                pts[j,1] = -cheba[a+1]
                pts[U+1-j,2] = -chebb[2d+2-b]
                j += 1
            end
        end
    end

    # evaluations
    u = calc_u(2, d, pts)
    P0 = Matrix{Float64}(undef, U, L)
    P0[:,1] .= 1
    col = 1
    for t in 1:d
        for xp in Combinatorics.multiexponents(2, t)
            col += 1
            P0[:,col] .= u[1][:,xp[1]+1] .* u[2][:,xp[2]+1]
        end
    end
    P = Array(qr(P0).Q)

    # weights TODO

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

    return (L=L, U=U, pts=pts, P0=P0, P=P, w=NaN)
end


function approxfekete_data(n::Int, d::Int)
    @assert d > 1
    @assert n > 1
    L = binomial(n+d, n)
    U = binomial(n+2d, n)

    # points in the initial interpolation grid
    npts = prod(2d+1:2d+n)
    _pts = Matrix{Float64}(undef, npts, n)
    for j in 1:n
        ig = prod(2d+1+j:2d+n)
        cs = cheb2_pts(2d+j)
        i = 1
        l = 1
        while true
            _pts[i:i+ig-1,j] .= cs[l]
            i += ig
            l += 1
            if l >= 2d+1+j
                if i >= npts
                    break
                end
                l = 1
            end
        end
    end

    # evaluations on the initial interpolation grid
    m = ones(U)
    u = calc_u(n, 2d, _pts)
    M = ones(npts, U)
    M[:,1] .= 1
    col = 1
    for t in 1:2d
        for xp in Combinatorics.multiexponents(n, t)
            col += 1
            for j in 1:n
                m[col] *= iseven(xp[j]) ? 2/(1 - xp[j]^2) : 0
                M[:,col] .*= u[j][:,xp[j]+1]
            end
        end
    end

    F = qr(M', Val(true))
    keep_pnt = F.p[1:U]

    w = F.R[:,1:U]\(F.Q'*m)
    pts = _pts[keep_pnt,:] # subset of points indexed with the support of w
    P0 = M[keep_pnt,1:L] # subset of polynomial evaluations up to total degree d
    P = Array(qr(P0).Q)

    return (L=L, U=U, pts=pts, P0=P0, P=P, w=w)
end
