module SaddleSearch

using Parameters

export run!


@with_kw type IterationLog
   numE::Vector{Int} = Int[]
   numdE::Vector{Int} = Int[]
   res_trans::Vector{Float64} = Float64[]
   res_rot::Vector{Float64} = Float64[]
end

function Base.push!(log::IterationLog, numE, numdE, res_trans, res_rot)
   push!(log.numE, numE)
   push!(log.numdE, numdE)
   push!(log.res_trans, res_trans)
   push!(log.res_rot, res_rot)
end


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

include("staticdimer.jl")

include("bbdimer.jl")

include("rotoptimdimer.jl")

include("superlineardimer.jl")

include("string.jl")

include("testsets.jl")

end # module
