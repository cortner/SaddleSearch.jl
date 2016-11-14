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


include("linesearch.jl")

include("staticdimer.jl")

include("bbdimer.jl")

include("rotoptimdimer.jl")

include("string.jl")

end # module
