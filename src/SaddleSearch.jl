module SaddleSearch

using Parameters

export run!


include("iterlog.jl")


Base.dot{T}(x, A::UniformScaling{T}, y) = A.λ * dot(x,y)
Base.dot(x, A::AbstractMatrix, y) = dot(x, A*y)



"""
An abstract linear operator representing `P + s * (Pv) ⊗ (Pv)`

Define `*` and `\`, the latter via Sherman-Morrison-Woodbury formula.
"""
type PreconSMW{T} <: AbstractMatrix{T}
   P       # an invertiable N x N matrix (probably spd)
   v       # a vector of length N
   Pv      # the vector P * v
   s::T    # see doc
   smw::T  # the SMW-factor
end

PreconSMW(P, v, s) = PreconSMW(P, v, P*v, s, s / (1.0 + s * dot(v, P, v)))

import Base: *, \, size
(*)(A::PreconSMW, x::AbstractVector) = A.P * x + (A.s * dot(A.Pv, x)) * A.Pv
(\)(A::PreconSMW, f::AbstractVector) = (A.P \ f) - ((A.smw * dot(A.v, f)) * A.v)

include("linesearch.jl")

include("ode.jl")

include("dimer.jl")

include("bbdimer.jl")

include("rotoptimdimer.jl")

include("superlineardimer.jl")

include("string.jl")

include("neb.jl")

include("varstepstring.jl")

include("preconstring.jl")


include("odestring.jl")

include("odeneb.jl")

include("testsets.jl")

end # module
