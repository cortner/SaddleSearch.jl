
using Dierckx
export PreconStringMethod

"""
`PreconStringMethod`: a preconditioning variant of the string method.

### Parameters:
* 'precon_scheme' : preconditioning method
* `alpha` : step length
* `tol_res` : residual tolerance
* `maxnit` : maximum number of iterations
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
"""
@with_kw type PreconStringMethod
   precon_scheme = coordTransform()
   alpha::Float64
   refine_points::Int
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   # precon = I
   # precon_prep! = (P, x) -> P
   verbose::Int = 2
   # precon_cond::Bool = false
end

function run!{T}(method::PreconStringMethod, E, dE, x0::Vector{T}, t0::Vector{T})
   # read all the parameters
   @unpack precon_scheme, alpha, refine_points, tol_res, maxnit,
            verbose = method
   @unpack precon, precon_prep!, precon_cond, tangent_norm, gradDescent⟂, force_eval, maxres = precon_scheme
   # initialise variables
   x, t = copy(x0), copy(t0)
   param = collect(linspace(.0, 1., length(x)))
   Np = length(precon); P = i -> precon[mod(i-Np+1,Np)+1]
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
      precon = precon_prep!(precon, x)
      P = i -> precon[mod(i-1,Np)+1]
      t ./= [tangent_norm(P(i), t[i]) for i=1:length(x)]
      t[1] =zeros(t[1]); t[end]=zeros(t[1])

      # evaluate gradients
      dE0 = [dE(x[i]) for i=1:length(x)]
      dE0⟂ = gradDescent⟂(P, dE0, t)
      F = [force_eval(P(i), dE0[i], dE0⟂[i], t[i]) for i=1:length(x)]
      Fref = force_eval(P, dE0ref, t)
      numE += length(x); numdE += length(x)

      # perform linesearch to find optimal step
      # steps = []
      # ls = Backtracking(c1 = .2, mindecfact = 1.)
      # for i=1:length(x)
      #    push!(steps, linesearch!(ls, E, E0[i][], dot(dE0[i],-F[i]), x[i], -F[i], copy(alpha)))
      # end
      # α = [steps[i][1] for i=1:length(steps)]
      # for k=1:5
      #    α = [.5 * (α[1] + α[2]); [(.25 * (α[n-1] + α[n+1]) + .5 * α[n]) for n=2:length(α)-1]; (.5 * (α[end-1] + α[end]))]
      # end
      # numE += sum([steps[i][2] for i=1:length(steps)])

      # residual, store history
      res = maxres(P, dE0⟂, F)
      push!(log, numE, numdE, res)
      if verbose >= 2
         @printf("%4d |   %1.2e\n", nit, res)
      end
      if res <= tol_res
         if verbose >= 1
            println("PreconStringMethod terminates succesfully after $(nit) iterations")
         end
         return x, log
      end
      x -= alpha .* F

      # reparametrise
      x, t = reparametrise!(method, x, t, param, precon_scheme)

      # string refinement
      if refine_points > 0
         refine!(param, x, t, refine_points)
         x, t = reparametrise!(method, x, t, param, precon_scheme)
      end

   end
   if verbose >= 1
      println("PreconStringMethod terminated unsuccesfully after $(maxnit) iterations.")
   end
   return x, log
end

function reparametrise!(method::PreconStringMethod, x, t, param, precon_scheme)
   @unpack precon, precon_prep!, precon_cond = precon_scheme

   precon = precon_prep!(precon, x)
   Np = length(precon); P = i -> precon[mod(i-1,Np)+1]

   ds = [sqrt(dot(x[i+1]-x[i], (P(i)+P(i+1))/2, x[i+1]-x[i])) for i=1:length(x)-1]
   param_temp = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   param_temp /= param_temp[end]; param_temp[end] = 1.
   S = [Spline1D(param_temp, [x[j][i] for j=1:length(x)],
         w =  ones(length(x)), k = 3, bc = "error") for i=1:length(x[1])]
   x = [[S[i](s) for i in 1:length(S)] for s in param]
   t = [[derivative(S[i], s) for i in 1:length(S)] for s in param]
   return x, t
end
