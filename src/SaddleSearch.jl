module SaddleSearch

using Reexport

@reexport using LinearAlgebra, Printf, Dates

@reexport using Parameters

# include("abstractions.jl")

# logging, preconditioner transformation, weidth norms and dots,
include("misc.jl")

# some line-search related methods
include("linesearch.jl")

# numerical ode dolvers
include("ode.jl")

# =========== Walker-type saddle search methods ============

include("dimer_types.jl")

# dimer, bb, ode
include("dimer.jl")

include("superlineardimer.jl")


# ============ Sting type methods ===================
# string, neb type definitions
include("string_types.jl")

# path manipulation: linear algebra in the path's framework
include("pathmisc.jl")

include("momentum_descent_schemes.jl")

# string method
include("string.jl")

# NEB method
include("neb.jl")


end # module
