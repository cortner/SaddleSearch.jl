using Dierckx

function reparametrise!{T}(x::Vector{T}, t::Vector{T}, ds::T; parametrisation=linspace(0.,1.,length(x)) )

   s = [0; copy(ds)]; [s[i]+=s[i-1] for i=2:length(s)]
   s /= s[end]; s[end] = 1.

   S = [Spline1D(s, [x[i][j] for i=1:length(x)], w = ones(length(x)),
        k = 3, bc = "error") for j=1:length(x[1])]
   xref = [[Sj(s) for s in parametrisation] for Sj in S ]
   tref = [[derivative(Sj, s) for s in parametrisation] for Sj in S]

   x_ = cat(2, xref...)
   t_ = cat(2, tref...)
   [x[i] = x_[i,:] for i=1:length(x)]
   [t[i] = t_[i,:] for i=1:length(x)]
   return x, t
end


function refine!(param, refine_points, t)
   N = length(t)
   for n = 2:N-1
      cosine = dot(t[n-1], t[n+1]) / (norm(t[n-1]) * norm(t[n+1]))
      if ( cosine < 0 )
         n1 = n-1; n2 = n+1; k = refine_points
         k1 = floor(param[n1] * k); k2 = floor((param[end] - param[n2-1]) * k)
         k = k1 + k2
         s1 = (n1 - k1 == 1) ? [.0] : collect(linspace(.0, 1., n1 - k1 )) * param[n1]
         s2 = collect(param[n1] + linspace(.0, 1., k + 3 ) * (param[n2] - param[n1]))
         s3 = (N - n2 - k2 + 1 == 1) ? [1.] : collect(param[n2] + linspace(.0, 1., N - n2 - k2 + 1 ) * (1 - param[n2]))
         param[:] = [s1;  s2[2:end-1]; s3][:]
      else
         param[:] = collect(linspace(0., 1., length(t)))[:]
      end
   end
   return param
end
