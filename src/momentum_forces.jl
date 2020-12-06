# forcing term for momentum  descent NEB method
function central_diff_forces(X::Vector{Float64}, V::Vector{Float64}, path_type::Type{Path{T,NI}}, P, f, b, nit) where {T, NI}

   F, R, ndE, dotPx = f(X, P, nit)
   @show(R)
   # fx = 0.5 * X + V
   # fv  =  F - b * V
   fx = V
   fv  =  F - b * V
   Ftilde = [fx; fv]
   @show(norm(fx, Inf), norm(fv, Inf))

   return Ftilde, R, ndE, dot
end
