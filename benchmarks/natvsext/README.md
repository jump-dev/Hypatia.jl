# natural vs extended formulation comparisons

scripts in this directory are used for comparing _natural_ and _extended_
formulations for problems in the examples folder using Hypatia and Mosek

run scripts from this directory

## run script

this script spawns processes
puts raw output files into "raw" folder

killall julia; ~/julia/julia run.jl &> raw/bench.txt

## analyze script

puts analyzed output files into "analysis" folder

~/julia/julia analyze.jl
