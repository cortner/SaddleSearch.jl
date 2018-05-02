



# ========= OLD CODE BELOW ========

# using Dierckx
export StringMethod

"""
`StringMethod`: the most basic string method variant, minimising the energy
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
@with_kw type StringMethod
   alpha::Float64
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_cond::Bool = false
end


function run!{T}(method::StringMethod, E, dE, x0::Vector{T}, t0::Vector{T})
   # read all the parameters
   @unpack alpha, tol_res, maxnit,
            precon_prep!, verbose, precon_cond = method
   P=method.precon
   # initialise variables
   x, t = copy(x0), copy(t0)
   nit = 0
   numdE, numE = 0, 0
   log = PathLog()
   # and just start looping
   if verbose >= 2
      @printf("SADDLESEARCH:  time | nit |  sup|∇E|_∞   \n")
      @printf("SADDLESEARCH: ------|-----|-----------------\n")
   end
   for nit = 0:maxnit
      # normalise t
      P = precon_prep!(P, x)
      t ./= [norm(P, t[i]) for i=1:length(x)]
      t[1] =zeros(t[1]); t[end]=zeros(t[1])
      # evaluate gradients
      dE0 = [dE(x[i]) for i=1:length(x)]
      dE0⟂ = [P \ dE0[i] - dot(dE0[i],t[i])*t[i] for i = 1:length(x)]
      numdE += length(x)
      # residual, store history
      maxres = maximum([norm(P*dE0⟂[i],Inf) for i = 1:length(x)])
      push!(log, numE, numdE, maxres)
      if verbose >= 2
         dt = Dates.format(now(), "HH:MM")
         @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, maxres)
      end
      if maxres <= tol_res
         if verbose >= 1
            println("SADDLESEARCH: StringMethod terminates succesfully after $(nit) iterations")
         end
         return x, log
      end
      x -= alpha * dE0⟂
      # reparametrise
      ds = [norm(P, x[i+1]-x[i]) for i=1:length(x)-1]
      reparametrise!(x, t, ds)

   end
   if verbose >= 1
      println("SADDLESEARCH: StringMethod terminated unsuccesfully after $(maxnit) iterations.")
   end
   return x, log
end
