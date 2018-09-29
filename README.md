# Alfonso.jl
(under construction) unofficial Julia re-implementation of alfonso (ALgorithm FOr Non-Symmetric Optimization), originally by D. Papp and S. Yıldız: https://github.com/dpapp-github/alfonso

only works on Julia v0.7+

solves a pair of primal and dual cone programs

primal (over x,s):
```
  min  c'x :          duals
    b - Ax == 0       (y)
    h - Gx == s in K  (z)
```
dual (over z,y):
```
  max  -b'y - h'z :      duals
    c + A'y + G'z == 0   (x)
                z in K*  (s)
```
where K is a convex cone defined as a Cartesian product of recognized primitive cones, and K* is its dual cone

the primal-dual optimality conditions are
```
         b - Ax == 0
         h - Gx == s
  c + A'y + G'z == 0
            s'z == 0
              s in K
              z in K*
```
