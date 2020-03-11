#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for examples
=#

using Test
import Random
using LinearAlgebra

import Hypatia
import Hypatia.ModelUtilities
import Hypatia.Cones
import Hypatia.Models
import Hypatia.Solvers

abstract type InstanceSet end
struct MinimalInstances <: InstanceSet end
struct FastInstances <: InstanceSet end
struct SlowInstances <: InstanceSet end

abstract type ExampleInstance{T <: Real} end
