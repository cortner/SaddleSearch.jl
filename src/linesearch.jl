export StaticLineSearch, Backtracking



# ================================================

"""
`type StaticLineSearch`: does nothing, i.e., just returns the step
proposal.
"""
struct StaticLineSearch end

linesearch!(ls::StaticLineSearch, F, f0, df0, x, p, α; f0_goal=f0) = α, 0, nothing


"""
`type Backtracking`

kw-parameters:
* `c1`: Armijo constant
* `order`: order of interpolation during backtracking; currently only 2 is
   implemented
* `minα`: smallest allowed step (throws exception otherwise)
* `mindecfact`: minimal decrease factor
"""
@with_kw struct Backtracking
   c1::Float64 = 0.1
   order::Int = 2
   minα::Float64 = 1e-8
   mindecfact::Float64 = 0.25
end


function linesearch!(ls::Backtracking, F, f0, df0, x, p, α; f0_goal=f0, condition=iter->false)
   # read parameters, initialise variables
   @unpack c1, order, minα, mindecfact = ls
   @assert f0_goal >= f0
   fα = F(x + α * p)
   numE = 1
   # start backtracking loop
   while (fα > f0_goal + c1 * α * df0) && !condition(numE)
      # check which form of backtracking to use
      if order == 2
         α1 = - 0.5 * (df0 * α) / ((fα - f0)/α - df0)
         α = max(α1, α * mindecfact)     # avoid miniscule steps
      else
         error("currently only quadratic backtracking is implemented")
      end
      # check whether α is too small now; probably something has gone wrong
      if α < minα
         return NaN, numE, fα
      end
      # evaluate f again and loop
      fα = F(x + α * p)
      numE += 1
   end
   return α, numE, fα
end
