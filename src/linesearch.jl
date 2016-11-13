export StaticLineSearch, Backtracking

function rayleigh(v, x, len, E0, E, P)
   w = v / sqrt(dot(v, P, v))
   return 2.0 * (E(x+len/2*w) - 2.0 * E0 + E(x-len/2*v)) / len^2
end


localmerit(x, x0, v0, len, g0, λ0, E) = (
   0.5 * ( E(x+len/2*v0) + E(x-len/2*v0) )
   - 2.0 * dot(v0, g0) * dot(v0, x-x0)
   - λ0 * dot(v0, x-x0)^2  )


type StaticLineSearch end

linesearch!(ls::StaticLineSearch, F, f0, df0, x, p, α; f0_goal=f0) = α, 0, nothing


@with_kw type Backtracking
   c1::Float64 = 0.1
   order::Int = 2
   minα::Float64 = 1e-8
   mindecfact::Float64 = 0.25
end


function linesearch!(ls::Backtracking, F, f0, df0, x, p, α; f0_goal=f0)
   # read parameters, initialise variables
   @unpack c1, order, minα, mindecfact = ls
   @assert f0_goal >= f0
   fα = F(x + α * p)
   numE = 1
   # start backtracking loop
   while fα > f0_goal + c1 * α * df0
      # check which form of backtracking to use
      if order == 2
         α1 = - 0.5 * (df0 * α) / ((fα - f0)/α - df0)
         α = max(α1, α * mindecfact)     # avoid miniscule steps
      else
         error("currently only quadratic backtracking is implemented")
      end
      # check whether α is too small now; probably something has gone wrong
      @assert α > minα
      # evaluate f again and loop
      fα = F(x + α * p)
      numE += 1
   end
   return α, numE, fα
end
