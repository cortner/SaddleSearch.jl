
"""
`IterationLog` : store iteration information

### Usage Example:
```julia
ilog = IterationLog(:numE => Int, :numdE => Int, :res => Float64)
for n = 1:num_iter
   # do something with variables numE, numdE, res)
   push!(ilog, numE, numdE, res)
end
```
"""
type IterationLog
   keys::Tuple
   D::Dict
end

function IterationLog(args...)
   D = Dict()
   keys = tuple()
   for a in args
      if isa(a, Pair) && isa(a[2], DataType)
         keys = tuple(keys..., a[1])
         D[a[1]] = Vector{a[2]}()
      else
         keys = tuple(keys..., a)
         D[a] = Vector{Any}()
      end
   end
   return IterationLog(keys, D)
end

function Base.push!(log::IterationLog, args...)
   @assert length(args) == length(log.keys)
   for (k, a) in zip(log.keys, args)
      push!(log.D[k], a)
   end
   return log
end

"""
`DimerLog()` : generates an `IterationLog` for walker-type saddle search
methods
"""
DimerLog() = IterationLog(:numE => Int, :numdE => Int,
                          :res_trans => Float64, :res_rot => Float64)

"""
`PathLog()` : generates an `IterationLog` for string-type saddle search
methods
"""
PathLog() = IterationLog(:numE => Int, :numdE => Int, :maxres => Float64)

Base.getindex(l::IterationLog, idx) = l.D[idx]

# convenient access
numE(l::IterationLog) = l[:numE]
numdE(l::IterationLog) = l[:numdE]
res_trans(l::IterationLog) = l[:res_trans]
res_rot(l::IterationLog) = l[:res_rot]
maxres(l::IterationLog) = l[:maxres]




Base.dot{T}(x, A::UniformScaling{T}, y) = A.λ * dot(x,y)
Base.dot(x, A::AbstractMatrix, y) = dot(x, A*y)
Base.norm(P, x) = sqrt(dot(x, P*x))
dualnorm(P, f) = sqrt(dot(f, P \ f))

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
