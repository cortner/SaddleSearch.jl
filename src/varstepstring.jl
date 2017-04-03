
using Dierckx
export VarStepStringMethod

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
@with_kw type VarStepStringMethod
   alpha::Float64
   ls::Backtracking(c1 = .2, mindecfact = 1.)
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
   @unpack alpha, ls, refine_points, tol_res, maxnit,
            precon_prep!, verbose, precon_cond = method
   P=method.precon
   # initialise variables
   x, t = copy(x0), copy(t0)
   param = collect(linspace(.0, 1., length(x[1])))
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
      t ./= [sqrt(dot(t[i], P, t[i])) for i=1:length(x)]
      t[1] =zeros(t[1]); t[end]=zeros(t[1])

      # evaluate gradients
      E0  = [E(x[i]) for i=1:length(x)]
      dE0 = [dE(x[i]) for i=1:length(x)]
      dE0perp = [P \ dE0[i] - dot(dE0[i],t[i])*t[i] for i = 1:length(x)]
      numE += length(x); numdE += length(x)

      # perform linesearch to find optimal step
      steps = []
      push!(steps, linesearch!(ls, E, E0[i], dot(dE0[i],-dE0perp[i]), x[i], -dE0perp[i], copy(alpha))
      α = [steps[i][1] for i=1:length(steps)]
      for k=1:5
         α1 = copy(α)
         α1[1] = [.5 * (α[1] + α[2]); [(.25 * (α[n-1] + α[n+1]) + .5 * α[n]) for n=2:length(α)-1]; (.5 * (α[end-1] + α[end]))]
         α = α1
      end
      numE += sum([steps[i][2] for i=1:length(steps)])

      # residual, store history
      maxres = maximum([norm(dE0perp[i],Inf) for i = 1:length(x)])
      push!(log, numE, numdE, maxres)
      if verbose >= 2
         @printf("%4d |   %1.2e\n", nit, maxres)
      end
      if maxres <= tol_res
         if verbose >= 1
            println("StringMethod terminates succesfully after $(nit) iterations")
         end
         return x, log
      end
      x -= α .* dE0perp

      # reparametrise
      param, x, t = reparametrise!(x, t, P, param)

      # string refinement
      if refine_points > 0
         refine!(param, x, t, P, refine_points)
         x, t = reparametrise!(x, t, P, param)
      end

   end
   if verbose >= 1
      println("StringMethod terminated unsuccesfully after $(maxnit) iterations.")
   end
   return x, log
end

function reparametrise!(x, t, P, param)
   ds = [sqrt(dot(x[i+1]-x[i], P, x[i+1]-x[i])) for i=1:length(x)-1]
   param_temp = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   param_temp /= param_temp[end]; param_temp[end] = 1.
   S = [Spline1D(param_temp, [x[j][i] for j=1:length(param_temp)],
         w =  ones(length(x)), k = 3, bc = "error") for i=1:length(x[1])]
   x = [[S[i](s) for i in 1:length(S)] for s in param ]
   t = [[derivative(S[i], s) for i in 1:length(S)] for s in param]
   return x, t
end

function refine!(param, x, t, refine_points)
   N = length(x)
   for n = 2:N-1
      cosine = dot(t[n-1], t[n+1]) /(norm(t[n-1]) * norm(t[n+1]))
      if ( cosine < 0 )
         n1 = n-1; n2 = n+1; k = refine_points
         k1 = floor(s[n1] * k); k2 = floor((s[end] - s[n2-1]) * k)
         k = k1 + k2
         s1 = (n1 - k1 == 1) ? [.0] : collect(linspace(.0, 1., n1 - k1 )) * s[n1]
         s2 = collect(t[n1] + linspace(.0, 1., k + 3 ) * (s[n2] - s[n1]))
         s3 = (N - n2 - k2 + 1 == 1) ? [1.] : collect(s[n2] + linspace(.0, 1., N - n2 - k2 + 1 ) * (1 - s[n2]))
         param = [s1;  s2[2:end-1]; s3]
      else
         param = collect(linspace(.0, 1., N))
      end
   end
   return param, x, t
end
