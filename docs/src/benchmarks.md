# Benchmarking scripts

In the [benchmarks folder](https://github.com/chriscoey/Hypatia.jl/tree/master/benchmarks), we have scripts for running performance benchmarks on instances derived from our examples.
See [Examples](@ref) for descriptions of our examples, and see the [examples folder](https://github.com/chriscoey/Hypatia.jl/tree/master/examples) for the instances, which are generated on-the-fly.
The two types of benchmarks we run are:

  - natural versus extended formulations, for a selection of examples; see [natvsext](https://github.com/chriscoey/Hypatia.jl/tree/master/benchmarks/natvsext),
  - selected algorithmic options for the stepping procedures; see [stepper](https://github.com/chriscoey/Hypatia.jl/tree/master/benchmarks/stepper).
  
We also provide scripts to analyze the raw results output from the run scripts.
See the READMEs for simple instructions for running these scripts on your machines.
Each run script typically takes several days to complete if running all examples, and the analysis scripts only take seconds.
