
using Optim, Dierckx
export NudgedElasticBandMethod

"""
`NudgedElasticBandMethod`: the basic neb method variant, minimising the
energy normally to the string by successive steepest descent minimisations at
fixed step-sizes and constraining nodes by a tangential force applied to
adjacent nodes.

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
   k::Float64
   scheme::Symbol
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_cond::Bool = false
end


function run!{T}(method::NudgedElasticBandMethod, E, dE, x0::Vector{T})
      # read all the parameters
      @unpack alpha, k, scheme, tol_res, maxnit,
               precon_prep!, verbose, precon_cond = method
      P=method.precon
      # initialise variables
      x = copy(x0)
      nit = 0
      numdE, numE = 0, 0
      log = PathLog()
      # and just start looping
      if verbose >= 2
         @printf(" nit |  sup|∇E|_∞   \n")
         @printf("-----|-----------------\n")
      end
      for nit = 0:maxnit
         P = precon_prep!(P, x)
         # evaluate gradients, and more stuff
         N = length(x)
         dE0 = [dE(x[i]) for i=1:N]
         numdE += 1
         # evaluate the tangent and spring force along the path
         dxds=[]; Fk=[]
         if method.scheme == :simple
            # forward and central finite differences
            dxds = [(x[i+1]-x[i]) for i=2:N-1]
            dxds ./= [norm(dxds[i]) for i=1:length(dxds)]
            dxds = [ [zeros(dxds[1])]; dxds; [zeros(dxds[1])] ]
            Fk = k*N*N*[dot(x[i+1] - 2*x[i] + x[i-1], dxds[i]) * dxds[i] for i=2:N-1]
         elseif method.scheme == :central
            # forward and central finite differences
            dxds = [(x[i+1]-x[i-1])/2 for i=2:N-1]
            dxds ./= [norm(dxds[i]) for i=1:length(dxds)]
            dxds = [ [zeros(dxds[1])]; dxds; [zeros(dxds[1])] ]
            Fk = k*N*N*[dot(x[i+1] - 2*x[i] + x[i-1], dxds[i]) * dxds[i] for i=2:N-1]
         elseif method.scheme == :upwind
            # upwind scheme
            E0 = [E(x[i]) for i=1:N]; numE += 1
            ΔE0 = [E0[i-1]-E0[i] for i=2:N]
            index1 = [ΔE0[i]/abs(ΔE0[i]) - ΔE0[i+1]/abs(ΔE0[i+1]) for i=1:N-2]
            index2 = [E0[i+1]-E0[i-1]/abs(E0[i+1]-E0[i-1]) for i=2:N-1]
            Ediffmax = [max(abs(ΔE0[i+1]) , abs(ΔE0[i])) for i=1:N-2]
            Ediffmin = [min(abs(ΔE0[i+1]) , abs(ΔE0[i])) for i=1:N-2]
            f_weight = 0.5*[(1 + index2[i])*Ediffmax[i] + (index2[i] - 1) * Ediffmin[i] for i=1:N-2]
            b_weight = 0.5*[(1 + index2[i])*Ediffmin[i] + (index2[i] - 1) *     Ediffmax[i] for i=1:N-2]
            dxds = [(1 - index1[i-1]) .* f_weight[i-1] .* (x[i+1]-x[i]) + (1 + index1[i-1]) .* b_weight[i-1] .* (x[i]-x[i-1]) for i=2:N-1]
            dxds ./= [norm(dxds[i]) for i=1:length(dxds)]
            dxds = [ [zeros(dxds[1])]; dxds; [zeros(dxds[1])] ]
            Fk = k*N*N*[(norm(x[i+1]-x[i]) - norm(x[i]-x[i-1])) * dxds[i] for i=2:N-1]
         elseif method.scheme == :splines
            # spline scheme
            ds = [sqrt(dot(x[i+1]-x[i], x[i+1]-x[i])) for i=1:length(x)-1]
            s = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
            s /= s[end]; s[end] = 1.
            S = spline(s, x)
            dxds = [[derivative(S[i], si) for i in 1:length(S)] for si in s ]
            dxds ./= [norm(dxds[i]) for i=1:length(dxds)]
            dxds[1] =zeros(dxds[1]); dxds[end]=zeros(dxds[1])
            d²xds² = [[derivative(S[i], si, nu=2) for i in 1:length(S)] for si in s ]
            Fk = k*[dot(d²xds²[i],dxds[i]) * dxds[i] for i=2:N-1]
         else
            error("unknown differentiation scheme")
         end

         Fk = [[zeros(x[1])]; Fk; [zeros(x[1])] ]
         dE0⟂ = [dE0[i] - dot(dE0[i],dxds[i])*dxds[i] for i = 1:length(x)]

         # residual, store history
         maxres = maximum([norm(dE0⟂[i],Inf) for i = 1:length(x)])
         push!(log, numE, numdE, maxres)
         if verbose >= 2
            @printf("%4d |   %1.2e\n", nit, maxres)
         end
         if maxres <= tol_res
            if verbose >= 1
               println("NudgedElasticBandMethod terminates succesfully after $(nit) iterations")
            end
            return x, log
         end
         x -= alpha * ( dE0⟂ - Fk )
      end
      if verbose >= 1
         println("NudgedElasticBandMethod terminated unsuccesfully after $(maxnit) iterations.")
      end
      return x, log
   end

   spline_i(x, y, i) =  Spline1D( x, [y[j][i] for j=1:length(y)],
                                       w = ones(length(x)), k = 3, bc = "error" )
   spline(x,y) = [spline_i(x,y,i) for i=1:length(y[1])]