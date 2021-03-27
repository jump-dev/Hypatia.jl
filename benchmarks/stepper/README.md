# stepper option comparisons

scripts in this directory are used for comparing solver options in Hypatia

run scripts from this directory

## run script

this script does not spawn processes
puts raw output files into "raw" folder

killall julia; ~/julia/julia run.jl &> raw/bench.txt

## analyze script

puts analyzed output files into "analysis" folder

~/julia/julia analyze.jl
