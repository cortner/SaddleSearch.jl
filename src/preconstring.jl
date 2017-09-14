
using Dierckx
export PreconStringMethod

"""
`PreconStringMethod`: a preconditioning variant of the string method.

### Parameters:
* 'precon_scheme' : preconditioning method
* `alpha` : step length
* `refine_points` : number of points allowed in refinement region, negative for no refinement of path
* `ls_cond` : true/false whether to perform linesearch during the minimisation step
* `tol_res` : residual tolerance
* `maxnit` : maximum number of iterations
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
"""
@with_kw type PreconStringMethod
   precon_scheme = coordTransform()
   alpha::Float64
   refine_points::Int = -1
   ls_cond::Bool = false
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
   @unpack precon_scheme, alpha, refine_points, ls_cond, tol_res, maxnit,
            verbose = method
   @unpack precon, precon_prep!, precon_cond, tangent_norm, gradDescent⟂, force_eval, maxres = precon_scheme
   # initialise variables
   x, t = copy(x0), copy(t0)
   param = linspace(.0, 1., length(x)) |> collect
   Np = length(precon)
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
      numdE += length(x)

      # evaluate force term
      dE0⟂ = gradDescent⟂(P, dE0, t)
      F = force_eval(P, dE0, dE0⟂, t)

      # perform linesearch to find optimal step
      if ls_cond
         E0  = [E(x[i]) for i=1:length(x)]
         numE += length(x)
         α = []
         ls = Backtracking(c1 = .2, mindecfact = .1, minα = 0.)
         for i=1:length(x)
            αi, cost, _ = linesearch!(ls, E, E0[i], dot(dE0[i],-F[i]), x[i], -F[i], copy(alpha), condition=iter->iter>=10)
            push!(α, αi)
            numE += cost
         end

         for k=1:10
            α = [.5 * (α[1] + α[2]); [(.25 * (α[n-1] + α[n+1]) + .5 * α[n]) for n=2:length(α)-1]; (.5 * (α[end-1] + α[end]))]
         end
      else
         α = alpha
      end

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
      x -= α .* F

      # reparametrise
      ds = [norm( 0.5*(P(i)+P(i+1)), x[i+1]-x[i] ) for i=1:length(x)-1]
      reparametrise!(x, t, ds, parametrisation = param)

      # string refinement
      if refine_points > 0
         refine!(param, refine_points, t)
         ds = [norm( 0.5*(P(i)+P(i+1)), x[i+1]-x[i] ) for i=1:length(x)-1]
         reparametrise!(x, t, ds, parametrisation = param)
      end

   end
   if verbose >= 1
      println("PreconStringMethod terminated unsuccesfully after $(maxnit) iterations.")
   end
   return x, log
end
