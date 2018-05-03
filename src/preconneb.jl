
using Optim, Dierckx
export PreconNudgedElasticBandMethod

"""
`PreconNudgedElasticBandMethod`: a preconditioned neb variant

### Parameters:
* `precon_scheme` : local/global preconditioning
* `alpha` : step length
* `k` : spring constant
* `scheme` : finite difference scheme used to evaluate gradients
* `refine_points` : -1 for no slope tracking and path refinement
* `ls_cond` : true/false whether to perform line search to find next time step
* `tol_res` : residual tolerance
* `maxnit` : maximum number of iterations
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
"""
@with_kw type PreconNudgedElasticBandMethod
   precon_scheme = localPrecon()
   alpha::Float64
   k::Float64
   scheme::Symbol
   refine_points::Int = -1
   ls_cond::Bool = false
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   verbose::Int = 2
end


function run!{T}(method::PreconNudgedElasticBandMethod, E, dE, x0::Vector{T})
   # read all the parameters
   @unpack precon_scheme, alpha, k, scheme, refine_points, ls_cond, tol_res,
            maxnit, verbose = method
   @unpack precon, precon_prep!, precon_cond, dist, point_norm,
            proj_grad, forcing, elastic_force, maxres = precon_scheme
   # initialise variables
   x = copy(x0)
   param = linspace(.0, 1., length(x)) |> collect
   Np = size(precon, 1); N = length(x)
   nit = 0
   numdE, numE = 0, 0
   log = PathLog()
   # and just start looping
   if verbose >= 2
      @printf("SADDLESEARCH:  time | nit |  sup|∇E|_∞   \n")
      @printf("SADDLESEARCH: ------|-----|-----------------\n")
   end
   for nit = 0:maxnit
      precon = precon_prep!(precon, x)
      function P(i) return precon[mod(i-1,Np)+1, 1]; end
      function P(i, j) return precon[mod(i-1,Np)+1, mod(j-1,Np)+1]; end

      # evaluate gradients
      dE0 = [dE(x[i]) for i=1:N]
      numdE += length(x)

      # evaluate the tangent and spring force along the path
      dxds=[]; Fk=[]
      if scheme == :simple
         # forward and central finite differences
         dxds = [ [zeros(x[1])]; [(x[i+1]-x[i]) for i=2:N-1]; [zeros(x[1])] ]
         dxds ./= point_norm(P, dxds)
         d²xds² = [ [zeros(x[1])]; [x[i+1] - 2*x[i] + x[i-1] for i=2:N-1];
                                                               [zeros(x[1])] ]
         # k *= N*N
      elseif scheme == :central
         # central finite differences
         dxds = [ [zeros(x[1])]; [0.5*(x[i+1]-x[i-1]) for i=2:N-1];
                                                               [zeros(x[1])] ]
         dxds ./= point_norm(P, dxds)
         # dxds = [ [zeros(x[1])]; dxds; [zeros(dxds[1])] ]
         d²xds² = [ [zeros(x[1])]; [x[i+1] - 2*x[i] + x[i-1] for i=2:N-1];
                                                               [zeros(x[1])] ]
         # k *= N*N
      elseif scheme == :upwind
         # upwind scheme
         E0 = [E(x[i]) for i=1:N]; numE += length(x)
         ΔE0 = [E0[i-1]-E0[i] for i=2:N]
         index1 = [ΔE0[i]/abs(ΔE0[i]) - ΔE0[i+1]/abs(ΔE0[i+1]) for i=1:N-2]
         index2 = [E0[i+1]-E0[i-1]/abs(E0[i+1]-E0[i-1]) for i=2:N-1]
         Ediffmax = [maximum([abs(ΔE0[i+1]) abs(ΔE0[i])], 2) for i=1:N-2]
         Ediffmin = [minimum([abs(ΔE0[i+1]) abs(ΔE0[i])], 2) for i=1:N-2]
         f_weight = 0.5*[(1 + index2[i])*Ediffmax[i] + (index2[i] - 1) * Ediffmin[i] for i=1:N-2]
         b_weight = 0.5*[(1 + index2[i])*Ediffmin[i] + (index2[i] - 1) *     Ediffmax[i] for i=1:N-2]
         dxds = [(1 - index1[i-1]) .* f_weight[i-1] .* (x[i+1]-x[i]) + (1 + index1[i-1]) .* b_weight[i-1] .* (x[i]-x[i-1]) for i=2:N-1]
         dxds = [ [zeros(dxds[1])]; dxds; [zeros(dxds[1])] ]
         dxds ./= point_norm(P, dxds)
         d²xds² = [ [zeros(x[1])]; [x[i+1] - 2*x[i] + x[i-1] for i=2:N-1]; [zeros(x[1])] ]
         # k *= N*N
      elseif scheme == :splines
         # spline scheme
         ds = [norm( 0.5*(P(i)+P(i+1)), x[i+1]-x[i] ) for i=1:length(x)-1]

         s = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
         s /= s[end]; s[end] = 1.
         S = [Spline1D(s, [x[j][i] for j=1:length(s)], w = ones(length(x)),
               k = 3, bc = "error") for i=1:length(x[1])]
         dxds = [[derivative(S[i], si) for i in 1:length(S)] for si in s ]
         dxds ./= point_norm(P, dxds)
         dxds[1] =zeros(dxds[1]); dxds[end]=zeros(dxds[1])
         d²xds² = [[derivative(S[i], si, nu=2) for i in 1:length(S)] for si in s]
         k *= (1/(N*N))
      else
         error("SADDLESEARCH: unknown differentiation scheme")
      end

      Fk = elastic_force(P, k*N*N, dxds, d²xds²)
      dE0⟂ = proj_grad(P, dE0, dxds)
      F = forcing(precon, dE0⟂-Fk); f = set_ref!(copy(x), F)

      # perform linesearch to find optimal step
      if ls_cond
         E0  = [E(x[i]) for i=1:length(x)]
         numE += length(x)
         α = []
         ls = Backtracking(c1 = .2, mindecfact = .1, minα = 0.)
         for i=1:length(x)
            αi, cost, _ = linesearch!(ls, E, E0[i], dot(dE0[i], f[i]), x[i], f[i], copy(alpha), condition=iter->iter>=10)
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
      res = maxres(P, dE0⟂)

      push!(log, numE, numdE, res)
      if verbose >= 2
         dt = Dates.format(now(), "HH:MM")
         @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, res)
      end
      if res <= tol_res
         if verbose >= 1
            println("SADDLESEARCH: PreconNudgedElasticBandMethod terminates succesfully after $(nit) iterations")
         end
         return x, log
      end
      x += α .* f
   end
   if verbose >= 1
      println("SADDLESEARCH: PreconNudgedElasticBandMethod terminated unsuccesfully after $(maxnit) iterations.")
   end
   return x, log
end
