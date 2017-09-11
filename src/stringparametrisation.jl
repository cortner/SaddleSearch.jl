using Dierckx

function reparametrise!{T}(method::StringMethod, x::Vector{T}, t::Vector{T})
   P=method.precon

   ds = [norm(P, x[i+1]-x[i]) for i=1:length(x)-1]
   s = [0; copy(ds)]; [s[i]+=s[i-1] for i=2:length(s)]
   s /= s[end]; s[end] = 1.

   S = [Spline1D(s, [x[i][j] for i=1:length(s)], w = ones(length(x)),
         k = 3, bc = "error") for j=1:length(x[1])]
   X = [[Sj(s) for s in linspace(0.,1.,length(x))] for Sj in S ]
   T = [[derivative(Sj, s) for s in linspace(0.,1.,length(x))] for Sj in S]

   Xref = cat(2, X...)
   Tref = cat(2, T...)
   [x[i] = Xref[i,:] for i=1:length(x)]
   [t[i] = Tref[i,:] for i=1:length(x)]
   return x, t
end
