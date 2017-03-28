
using Dierckx
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
   ls::Backtracking(c1 = .2, mindecfact = 1.)
   refine_param::Int
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
   @unpack alpha, ls, refine_param, tol_res, maxnit,
            precon_prep!, verbose, precon_cond = method
   P=method.precon
   # initialise variables
   x, t = copy(x0), copy(t0)
   parametrisation = collect(linspace(.0, 1., N))
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

      # evaluate gradients, and more stuff
      E0  = [E(x[i]) for i=1:length(x)]
      dE0 = [dE(x[i]) for i=1:length(x)]
      t[1] =zeros(t[1]); t[end]=zeros(t[1])
      dE0perp = [P \ dE0[i] - dot(dE0[i],t[i])*t[i] for i = 1:length(x)]
      numE += length(x); numdE += length(x)

      # perform linesearch to find optimal step
      steps = []
      push!(steps, linesearch!(ls, E, E0[i], dE0[i], x[i], -dE0[i], copy(alpha))
      α = [steps[i][1] for i=1:length(steps)]
      for k=1:5
         α1 = copy(α)
         α1[1] = [.5 * (α[1] + α[2]); (.25 * (α[n-1] + α[n+1]) + .5 * α[n]) for n=2:length(α)-1]; (.5 * (α[end-1] + α[end]))]
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
      parametrisation, x, t = reparametrise!(parametrisation, x, t, P)

      # string refinement
      if refine_param > 0
         for n = 2:length(x)-1
            cosine = dot(t[n-1], t[n+1]) /(norm(t[n-1]) * norm(t[n+1]))
            if ( cosine < 0 )
               n1 = n-1; n2 = n+1; k=copy(refine_param)
               k1 = floor(s[n1] * k)
               k2 = floor((s[end] - s[n2-1]) * k)
               k = k1 + k2
               s1 = (n1 - k1 == 1) ? [.0] : collect(linspace(.0, 1., n1 - k1 )) * s[n1]
               s2 = collect(t[n1] + linspace(.0, 1., k + 3  ) * (s[n2] - s[n1]))
               s3 = (N - n2 - k2 + 1 == 1) ? [1.] : collect(s[n2] + linspace(.0, 1., N - n2 - k2 + 1 ) *(1 - s[n2]))
               s = [s1;  s2[2:end-1]; s3]
            else
               s = collect(linspace(.0, 1., N))
            end
         end
         parametrisation, x, t = reparametrise!(parametrisation, x, t, P)
      end

   end
   if verbose >= 1
      println("StringMethod terminated unsuccesfully after $(maxnit) iterations.")
   end
   return x, log
end

function reparametrise!(parametrisation, x, t, P)
   ds = [sqrt(dot(x[i+1]-x[i], P, x[i+1]-x[i])) for i=1:length(x)-1]
   parametrisation1 = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   parametrisation1 /= parametrisation1[end]; parametrisation1[end] = 1.
   S = spline(parametrisation1, x)
   x = [[S[i](s) for i in 1:length(S)] for s in parametrisation ]
   t = [[derivative(S[i], s) for i in 1:length(S)] for s in parametrisation]
   return parametrisation, x, t
end

spline_i(x, y, i) =  Spline1D( x, [y[j][i] for j=1:length(y)],
                                    w = ones(length(x)), k = 3, bc = "error" )
spline(x,y) = [spline_i(x,y,i) for i=1:length(y[1])]
