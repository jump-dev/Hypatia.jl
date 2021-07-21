# natural vs extended formulation comparisons

Scripts in this directory are used for comparing _natural_ and _extended_
formulations for problems in the `examples/` folder using Hypatia and Mosek.
The following instructions should work from a Linux/macOS shell/terminal.

## install Julia and dependencies

Install the selected version of Julia (e.g. v1.7) from
https://julialang.org/downloads/.

Install the selected version of MOSEK (e.g. version 9) by following the
instructions at https://github.com/MOSEK/Mosek.jl.

Start Julia from the shell and enter Julia's pkg mode by typing `]`.
Install the selected version of Hypatia (e.g. v0.5.0) and script dependencies
and run update:
```julia
pkg> add Hypatia#v0.5.0
pkg> add Combinatorics CSV DataFrames DataStructures DelimitedFiles Distributions
pkg> add DynamicPolynomials ForwardDiff JuMP PolyJuMP Random SemialgebraicSets
pkg> add SpecialFunctions SumOfSquares Test Printf MosekTools Distributed
pkg> up
```

Exit Julia, and change directory to the benchmarks/natvsext folder:
```shell
cd ~/.julia/dev/Hypatia/benchmarks/natvsext
```

## run.jl script

This script spawns a process for each instance and each solver, one at a time.
It kills the process if a memory or time limit is reached (see the options at
the top of run.jl).
It puts output files into the `raw/` folder, specifically a csv file that is used
by the analysis script, and a txt file of script progress and solver printouts.

Start a GNU Screen from the shell by typing `screen`
(see https://www.gnu.org/software/screen/ for installation).

Run (from the benchmarks/natvsext directory):
```shell
killall julia; ~/julia/julia run.jl &> raw/bench.txt
```
If the script errors in the next few minutes, follow the error messages to debug,
or if that fails, try starting Julia and running:
```julia
julia> include("run.jl")
```
Follow any prompts or error instructions (e.g. to install missing packages).

If the script does not error after a few minutes, detach from the GNU Screen
session by typing `ctrl+a` then `d` (to later reattach, type `screen -r`).
Monitor the progress of the script by typing:
```shell
cat raw/bench.txt
tail -f raw/bench.txt
```
The script should take several days to finish.

## analyze.jl script

This script reads the csv output of the run.jl script, `raw/bench.csv`, and
analyzes these results, outputting analysis/summaries into files in the
`analysis/` folder and printing some information to the shell.

Start Julia (from the benchmarks/natvsext directory) and run:
```julia
julia> include("analyze.jl")
```
Follow any error messages to debug.
The script should take at most a couple of minutes.
When finished, inspect the files in the `analysis/` folder.
