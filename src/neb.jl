
using Optim, Dierckx
export NudgedElasticBandMethod

"""
`NudgedElasticBandMethod`: the basic neb method variant, minimising the
energy normally to the string by successive steepest descent minimisations at
fixed step-sizes and constraining nodes by a tangential force applied to
adjacent nodes. The tangents were evaluated with the upwind scheme.

### Parameters:
* `alpha` : step length
* `k` : spring constant
* `tol_res` : residual tolerance
* `maxnit` : maximum number of iterations
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_cond` : true/false whether to precondition the minimisation step
"""
@with_kw type NudgedElasticBandMethod
   alpha::Float64
   k::Vector{Float64}
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_cond::Bool = false
end


function run!{T}(method::StringMethod, E, dE, x0::Vector{T})
      # read all the parameters
      @unpack alpha, k, tol_res, maxnit,
               precon_prep!, verbose, precon_cond = method
      P=method.precon
      # initialise variables
      x = copy(x0)
      nit = 0
      numdE, numE = 0, 0
      log = IterationLog()
      # and just start looping
      if verbose >= 2
         @printf(" nit |  sup|∇E|_∞   \n")
         @printf("-----|-----------------\n")
      end
      for nit = 0:maxnit
         P = precon_prep!(P, x)
         # evaluate gradients, and more stuff
         dE0 = [dE(x[i]) for i=1:length(x)]
         E0 = [E(x[i]) for i=1:length(x)]
         numdE += 1
         numE += 1

        #  ds = [sqrt(dot(x[i+1]-x[i], P, x[i+1]-x[i])) for i=1:length(x)-1]
        #  s = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
        #  s /= s[end]; s[end] = 1.
        #  S = spline(s, x)
        #  dxds = [[derivative(S[i], si) for i in 1:length(S)] for si in s]
        #  d²xds² = [[derivative(S[i], si, nu=2) for i in 1:length(S)] for si in s]
         # evaluate the tangents
         ΔE0 = [E0[i-1]-E0[i] for i=2:N];
         index1 = [ΔE0[i]/abs(ΔE0[i]) - ΔE0[i+1]/abs(ΔE0[i+1]) for i=1:N-1]
         index2 = [E0[i+1]-E0[i-1]/abs(E0[i+1]-E0[i-1]) for i=2:N];
         Ediffmax = [maximum(abs(ΔE0[i+1]) - abs(ΔE0[i])) for i=1:N-1]
         Ediffmin = [minimum(abs(ΔE0[i+1]) - abs(ΔE0[i])) for i=1:N-1]
         l_wind = [0.5*((1 + index2[i])*Ediffmax[i] + (index2[i] - 1)*Ediffmin[i]) for i=1:N-1]
         r_wind = [0.5*((1 + index2[i])*Ediffmin[i] + (index2[i] - 1)*Ediffmax[i]) for i=1:N-1]
         dxds = [(1 - index1) * l_wind[i] * (x[i+1]-x[i]) +
                  (1 + index1) * r_wind[i] * (x[i]-x[i-1]) for i=2:N-1]
         dxds = ./= [norm(dxds[i]) for i=1:length(dxds)]
         dxds = [ [zeros(dxds[1])]; dxds; [zeros(dxds[1])] ]

         # d²xds² = [(x[i+1]-x[i]) - (x[i]-x[i-1]) for i=2:N-1]
         # d²xds² = [[zeros(x[1])]; d²xds²; [zeros(x[1])]]

         # evaluate the spring force
         Fk = k * [(abs(x[i+1]-x[i]) - abs(x[i]-x[i-1])) * dxds[i] for i=2:N-1]
         Fk = [[zeros(dxds[1])]; Fk; [zeros(dxds[1])] ]
         dE0⟂ = [dE0[i] - dot(dE0[i],dxds[i])*dxds[i] for i = 1:length(x)]

         # residual, store history
         res = maximum([norm(dE0⟂[i],Inf) for i = 1:length(x)])
         push!(log, numE, numdE, res, 0)
         if verbose >= 2
            @printf("%4d |   %1.2e\n", nit, res)
         end
         if res <= tol_res
            if verbose >= 1
               println("NudgedElasticBandMethod terminates succesfully after $(nit) iterations")
            end
            return x, log
         end
         x -= alpha * ( dE0⟂ - Fk ) # k * dot(d²xds²,dxds) * dxds )
      end
      if verbose >= 1
         println("NudgedElasticBandMethod terminated unsuccesfully after $(maxnit) iterations.")
      end
      return x, log
   end
