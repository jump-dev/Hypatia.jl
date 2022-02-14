# stepper comparisons

Scripts in this directory are used for comparing stepping procedures in Hypatia
on problems in the `examples/` folder.
The following instructions should work from a Linux/macOS shell/terminal.

## install Julia and dependencies

Download the CBLIB instances (see https://cblib.zib.de/download/readme.txt)
for the CBLIB example (at `Hypatia/examples/CBLIB/JuMP_test.jl`) by running in
the shell (from the home directory):
```shell
cd ~
mkdir -p cblib
cd cblib
wget -r -l1 -np http://cblib.zib.de/download/all/ -A expdesign_D_8_4.cbf.gz,port_12_9_3_a_1.cbf.gz,tls4.cbf.gz,ck_n25_m10_o1_1.cbf.gz,rsyn0805h.cbf.gz,2x3_3bars.cbf.gz,HMCR-n20-m400.cbf.gz,classical_20_0.cbf.gz,achtziger_stolpe06-6.1flowc.cbf.gz,LogExpCR-n100-m400.cbf.gz
```
Check that these `.cbf.gz` files (and possibly some extra files that can be
ignored) are in a new folder `~/cblib/cblib.zib.de/download/all/`.

Install the selected version of Julia (e.g. v1.7) from
https://julialang.org/downloads/.

Start Julia (e.g. `~/julia/julia`) from the shell and enter Julia's pkg mode by typing `]`.
Install Hypatia and the script dependencies:
```julia
pkg> dev Hypatia
pkg> add Combinatorics CSV DataFrames DataStructures DelimitedFiles Distributions
pkg> add DynamicPolynomials ForwardDiff JuMP PolyJuMP Random SemialgebraicSets
pkg> add SpecialFunctions SumOfSquares Test Printf BenchmarkProfiles
```
Exit Julia.
Set the desired version of Hypatia (e.g. v0.5.2-patch) with:
```shell
cd ~/.julia/dev/Hypatia
git checkout v0.5.2-patch
```
Update packages by starting Julia again and typing `]`, then:
```julia
pkg> up
```
Exit Julia, and change directory to the benchmarks/stepper folder:
```shell
cd ~/.julia/dev/Hypatia/benchmarks/stepper
```

## run.jl script

This script does not spawn processes, as it should not encounter fatal errors and
should not require time limits.
It puts output files into the `raw/` folder, specifically a csv file that is used
by the analysis script, and a txt file of script progress and solver printouts.

Start a GNU Screen from the shell by typing `screen`
(see https://www.gnu.org/software/screen/ for installation).

Run (from the benchmarks/stepper directory):
```shell
mkdir -p raw
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
The script should take a day or two to finish.

## analyze.jl script

This script reads the csv output of the run.jl script, `raw/bench.csv`, and
analyzes these results, outputting analysis/summaries into files in the
`analysis/` folder and printing some information to the shell.

Start Julia (from the benchmarks/stepper directory) and run:
```julia
julia> include("analyze.jl")
```
Follow any error messages to debug.
The script should take at most a couple of minutes.
When finished, inspect the files in the `analysis/` folder.
