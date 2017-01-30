
"""
Usage Example:
```
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


DimerLog() = IterationLog(:numE => Int, :numdE => Int,
                  :res_trans => Float64, :res_rot => Float64)

Base.getindex(l::IterationLog, idx) = l.D[idx]

numE(l::IterationLog) = l[:numE]
numdE(l::IterationLog) = l[:numdE]
res_trans(l::IterationLog) = l[:res_trans]
res_rot(l::IterationLog) = l[:res_rot]
