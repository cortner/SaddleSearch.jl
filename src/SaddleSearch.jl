module SaddleSearch

using Parameters

export run!


# logging, preconditioner transformation, weidth norms and dots,
include("misc.jl")

# some line-search related methods
include("linesearch.jl")

include("ode.jl")

# =========== Walker-type saddle search methods ============

include("dimer_types.jl")

# dimer, bb, ode
include("dimer.jl")

include("superlineardimer.jl")


# ============ Sting type methods ===================
include("string_types.jl")

include("string.jl")

include("neb.jl")

include("pathmisc.jl")


end # module
