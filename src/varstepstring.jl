
using Dierckx
export VarStepStringMethod

"""
`VarStepStringMethod`: the most basic string method variant, minimising the energy
normally to the string by successive steepest descent minimisations at fixed
step-size with an intermediate redistribution of the nodes.

### Parameters:
* `alpha` : step length
* `tol_res` : residual tolerance
* `maxnit` : maximum number of iterations
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_cond` : true/false whether to precondition the minimisation step
"""
@with_kw type VarStepStringMethod
   alpha::Float64
   refine_points::Int = 3
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_cond::Bool = false
end


function run!{T}(method::VarStepStringMethod, E, dE, x0::Vector{T}, t0::Vector{T})
   # read all the parameters
   @unpack alpha, refine_points, tol_res, maxnit,
            precon_prep!, verbose, precon_cond = method
   P=method.precon
   # initialise variables
   x, t = copy(x0), copy(t0)
   param = linspace(.0, 1., length(x)) |> collect
   nit = 0
   numdE, numE = 0, 0
   log = PathLog()
   # and just start looping
   if verbose >= 2
      @printf(" nit |  sup|∇E|_∞   \n")
      @printf("-----|-----------------\n")
   end
   for nit = 0:maxnit
      # normalise t
      P = precon_prep!(P, x)
      t ./= [norm(P, t[i]) for i=1:length(x)]
      t[1] =zeros(t[1]); t[end]=zeros(t[1])

      # evaluate gradients
      E0  = [E(x[i]) for i=1:length(x)]
      dE0 = [dE(x[i]) for i=1:length(x)]
      dE0⟂ = [P \ dE0[i] - dot(dE0[i],t[i])*t[i] for i = 1:length(x)]
      numE += length(x); numdE += length(x)

      # perform linesearch to find optimal step
      α = []
      ls = Backtracking(c1 = .2, mindecfact = .1, minα = 0.)
      for i=1:length(x)
         αi, cost, _ = linesearch!(ls, E, E0[i], dot(dE0[i],-dE0⟂[i]), x[i], -dE0⟂[i], copy(alpha), condition=iter->iter>=10)
         push!(α, αi)
         numE += cost
      end

      for k=1:10
         α = [.5 * (α[1] + α[2]); [.25 * (α[n-1] + α[n+1]) + .5 * α[n] for n=2:length(α)-1]; .5 * (α[end-1] + α[end])]
      end

      # residual, store history
      maxres = maximum([norm(dE0⟂[i],Inf) for i = 1:length(x)])
      push!(log, numE, numdE, maxres)
      if verbose >= 2
         @printf("%4d |   %1.2e\n", nit, maxres)
      end
      if maxres <= tol_res
         if verbose >= 1
            println("VarStepStringMethod terminates succesfully after $(nit) iterations")
         end
         return x, log
      end
      x -= α .* dE0⟂

      # reparametrise
      ds = [norm(P, x[i+1]-x[i]) for i=1:length(x)-1]
      reparametrise!(x, t, ds, parametrisation = param)

      # string refinement
      if refine_points > 0
         refine!(param, refine_points, t)
         ds = [norm(P, x[i+1]-x[i]) for i=1:length(x)-1]
         reparametrise!(x, t, ds, parametrisation = param)
      end

   end
   if verbose >= 1
      println("VarStepStringMethod terminated unsuccesfully after $(maxnit) iterations.")
   end
   return x, log
end
