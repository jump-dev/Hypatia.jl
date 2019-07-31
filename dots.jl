# function symdot(A::Symmetric{T, Matrix{T}}, B::Symmetric{T, Matrix{T}}) where {T}
#     ret = zero(T)
#     m = size(A, 2)
#     @inbounds for j in 1:m
#         for i in 1:(j - 1)
#             ret += A[i, j] * B[i, j] * 2
#         end
#         ret += A[j, j] * B[j, j]
#     end
#     return ret
# end

# https://en.wikipedia.org/wiki/Frobenius_inner_product
function symdot(A::HermOrSym, B::HermOrSym)
    m = size(A, 2)
    if m != size(B, 2)
        throw(DimensionMismatch("first array has side length $(m) which does not match the length of the second, $(size(B, 2))."))
    end
    if m == 0
        return dot(zero(eltype(A)), zero(eltype(B)))
    end
    s = zero(dot(first(A).re, first(B).re))
    for j in 1:m
        for i in 1:(j - 1)
            @inbounds s += 2 * real(conj(A[i, j]) * B[i, j])
        end
        s += A[j, j] * B[j, j]
    end
    return s
end

@testset "hermitian dot" begin
    eltypes = (Float32, Float64, Int)
    for elty1 in eltypes, elty2 in eltypes

        AA = Hermitian(convert(Matrix{Complex{elty1}}, [1+2im 3+4im; 0 5+6im]))
        BB = Hermitian(convert(Matrix{Complex{elty2}}, [7+8im 9+10im; 0 11+12im]))
        @test symdot(AA, BB) == convert(elty, 196)

        AA = Symmetric(convert(Matrix{elty1}, [1 2; 0 3]))
        BB = Symmetric(convert(Matrix{elty2}, [4 5; 0 6]))
        @test symdot(AA, BB) == convert(elty, 42)

        AA = Hermitian(convert(Matrix{Complex{elty1}}, [1+2im 3+4im; 0 5+6im]))
        BB = Symmetric(convert(Matrix{elty2}, [7 8; 9 10]))
        @test symdot(AA, BB) == convert(elty, 105)
    end
end


using LinearAlgebra, Test

@test dot(Any[1.0,2.0], Any[3.5,4.5]) === 12.5

@testset "dot" for elty in (Float32, Float64, ComplexF32, ComplexF64)
    x = convert(Vector{elty},[1.0, 2.0, 3.0])
    y = convert(Vector{elty},[3.5, 4.5, 5.5])
    @test_throws DimensionMismatch dot(x, 1:2, y, 1:3)
    @test_throws BoundsError dot(x, 1:4, y, 1:4)
    @test_throws BoundsError dot(x, 1:3, y, 2:4)
    @test dot(x, 1:2, y, 1:2) == convert(elty, 12.5)
    @test transpose(x)*y == convert(elty, 29.0)
    X = convert(Matrix{elty},[1.0 2.0; 3.0 4.0])
    Y = convert(Matrix{elty},[1.5 2.5; 3.5 4.5])
    @test dot(X, Y) == convert(elty, 35.0)
    Z = convert(Vector{Matrix{elty}},[reshape(1:4, 2, 2), fill(1, 2, 2)])
    @test dot(Z, Z) == convert(elty, 34.0)
end





dot1(x,y) = invoke(dot, Tuple{Any,Any}, x,y)
dot2(x,y) = invoke(dot, Tuple{AbstractArray,AbstractArray}, x,y)
@testset "generic dot" begin
    AA = [1+2im 3+4im; 5+6im 7+8im]
    BB = [2+7im 4+1im; 3+8im 6+5im]
    for A in (copy(AA), view(AA, 1:2, 1:2)), B in (copy(BB), view(BB, 1:2, 1:2))
        @test dot(A,B) == dot(vec(A),vec(B)) == dot1(A,B) == dot2(A,B) == dot(float.(A),float.(B))
        @test dot(Int[], Int[]) == 0 == dot1(Int[], Int[]) == dot2(Int[], Int[])
        @test_throws MethodError dot(Any[], Any[])
        @test_throws MethodError dot1(Any[], Any[])
        @test_throws MethodError dot2(Any[], Any[])
        for n1 = 0:2, n2 = 0:2, d in (dot, dot1, dot2)
            if n1 != n2
                @test_throws DimensionMismatch d(1:n1, 1:n2)
            else
                @test d(1:n1, 1:n2) ≈ norm(1:n1)^2
            end
        end
    end
end



function dot(x::AbstractArray, y::AbstractArray)
    lx = length(x)
    if lx != length(y)
        throw(DimensionMismatch("first array has length $(lx) which does not match the length of the second, $(length(y))."))
    end
    if lx == 0
        return dot(zero(eltype(x)), zero(eltype(y)))
    end
    s = zero(dot(first(x), first(y)))
    for (Ix, Iy) in zip(eachindex(x), eachindex(y))
        @inbounds s += dot(x[Ix], y[Iy])
    end
    s
end


symdot(A::Symmetric{T, Matrix{T}}, B::Symmetric{T, Matrix{T}}) where {T} = @inbounds sum(A[j, j] * B[j, j] + 2 * sum(A[i, j] * B[i, j] for i in 1:(j - 1)) for j in 1:size(A, 2))


n = 5000
A = Symmetric(randn(n, n));
B = Symmetric(randn(n, n));
@benchmark dot(A, B)
@benchmark symdot(A, B)
