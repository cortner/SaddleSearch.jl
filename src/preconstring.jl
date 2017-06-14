
using Dierckx
export PreconStringMethod

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
@with_kw type PreconStringMethod
   alpha::Float64
   refine_points::Int
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_cond::Bool = false
end


function run!{T}(method::PreconStringMethod, E, dE, x0::Vector{T}, t0::Vector{T})
   # read all the parameters
   @unpack alpha, refine_points, tol_res, maxnit,
            precon_prep!, verbose, precon_cond = method
   P=method.precon
   # initialise variables
   x, t = copy(x0), copy(t0)
   param = collect(linspace(.0, 1., length(x)))
   Np = length(P)
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
      t ./= [sqrt(dot(t[i], P[mod(i-Np+1,Np)+1], t[i])) for i=1:length(x)]
      t[1] =zeros(t[1]); t[end]=zeros(t[1])

      # evaluate gradients
      E0  = [E(x[i]) for i=1:length(x)]
      dE0 = [dE(x[i]) for i=1:length(x)]
      dE0⟂ = [P[mod(i-Np+1,Np)+1] \ dE0[i] - dot(dE0[i],t[i])*t[i] for i = 1:length(x)]
      numE += length(x); numdE += length(x)

      # perform linesearch to find optimal step
      # steps = []
      # ls = Backtracking(c1 = .2, mindecfact = 1.)
      # for i=1:length(x)
      #    push!(steps, linesearch!(ls, E, E0[i][], dot(dE0[i],-dE0⟂[i]), x[i], -dE0⟂[i], copy(alpha)))
      # end
      # α = [steps[i][1] for i=1:length(steps)]
      # for k=1:5
      #    α = [.5 * (α[1] + α[2]); [(.25 * (α[n-1] + α[n+1]) + .5 * α[n]) for n=2:length(α)-1]; (.5 * (α[end-1] + α[end]))]
      # end
      # numE += sum([steps[i][2] for i=1:length(steps)])

      # residual, store history
      maxres = maximum([norm(P[mod(i-Np+1,Np)+1]*dE0⟂[i],Inf) for i = 1:length(x)])
      push!(log, numE, numdE, maxres)
      if verbose >= 2
         @printf("%4d |   %1.2e\n", nit, maxres)
      end
      if maxres <= tol_res
         if verbose >= 1
            println("PreconStringMethod terminates succesfully after $(nit) iterations")
         end
         return x, log
      end
      x -= alpha .* dE0⟂

      # reparametrise
      x, t = reparametrise!(method, x, t, P, param)

      # string refinement
      if refine_points > 0
         refine!(param, x, t, refine_points)
         x, t = reparametrise!(method, x, t, P, param)
      end

   end
   if verbose >= 1
      println("PreconStringMethod terminated unsuccesfully after $(maxnit) iterations.")
   end
   return x, log
end

function reparametrise!(method::PreconStringMethod, x, t, P, param)
   Np = length(P)
   ds = [sqrt(dot(x[i+1]-x[i], (P[mod(i-Np+1,Np)+1]+P[mod(i-1-Np+1,Np)+1])/2, x[i+1]-x[i])) for i=1:length(x)-1]
   param_temp = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   param_temp /= param_temp[end]; param_temp[end] = 1.
   S = [Spline1D(param_temp, [x[j][i] for j=1:length(x)],
         w =  ones(length(x)), k = 3, bc = "error") for i=1:length(x[1])]
   x = [[S[i](s) for i in 1:length(S)] for s in param]
   t = [[derivative(S[i], s) for i in 1:length(S)] for s in param]
   return x, t
end
