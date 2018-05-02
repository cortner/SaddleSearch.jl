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

include("string.jl")

include("neb.jl")

include("varstepstring.jl")

include("preconstring.jl")

include("preconneb.jl")

include("odestring.jl")

include("odeneb.jl")

include("pathpreconschemes.jl")

include("pathtraversing.jl")

include("stringparametrisation.jl")



end # module
