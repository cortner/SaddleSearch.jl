
using Dierckx
export ODENudgedElasticbandMethod

"""
`ODENudgedElasticbandMethod`: neb variant utilising adaptive time step ode solvers

### Parameters:
* `alpha` : step length
* 'k' : spring constrant
* 'abstol' : absolute errors tolerance
* 'reltol' : relative errors tolerance
* `tol_res` : residual tolerance
* `maxnit` : maximum number of iterations
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_cond` : true/false whether to precondition the minimisation step
"""
@with_kw type ODENudgedElasticBandMethod
   solver = ode12(1e-6, 1e-3, true)
   precon_scheme = localPrecon()
   k::Float64
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   # precon = [I]
   # precon_prep! = (P, x) -> P
   verbose::Int = 2
   # precon_cond::Bool = false
end


function run!{T}(method::ODENudgedElasticBandMethod, E, dE, x0::Vector{T})
   # read all the parameters
   @unpack solver, precon_scheme, k, tol_res, maxnit, verbose = method
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

   αout, xout, log = odesolve(solver, (α_,x_) -> forces(precon_scheme, x, x_, k,
   dE), ref(x), length(x), log, method; tol_res = tol_res, maxnit=maxnit )

   x = set_ref!(x, xout[end])
   return x, log, αout
end

function forces{T}(precon_scheme, x::Vector{T}, xref::Vector{Float64},
                     k::Float64, dE)
   x = set_ref!(x, xref)
   N = length(x)
   precon = precon_prep!(precon, x)
   Np = size(precon, 1); # P = i -> precon[mod(i-1,Np)+1]
   function P(i) return precon[mod(i-1,Np)+1, 1]; end
   function P(i, j) return precon[mod(i-1,Np)+1, mod(j-1,Np)+1]; end

   # central finite differences
   dxds = [[zeros(x[1])]; [0.5*(x[i+1]-x[i-1]) for i=2:N-1]; [zeros(x[1])]]
   dxds ./= point_norm(P, dxds) # [norm(P(i), dxds[i]) for i=1:length(dxds)]
   # dxds = [ [zeros(dxds[1])]; dxds; [zeros(dxds[1])] ]
   d²xds² = [ [zeros(x[1])]; [x[i+1] - 2*x[i] + x[i-1] for i=2:N-1];
                                                         [zeros(x[1])] ]
   Fk = elastic_force(P, k, dxds, d²xds²)
   #k*[dot(x[i+1] - 2*x[i] + x[i-1], P(i), dxds[i]) * dxds[i] for i=2:N-1]
   # Fk = [[zeros(x[1])]; Fk; [zeros(x[1])] ]

   dE0 = [dE(x[i]) for i=1:length(x)]

   dE0⟂ = proj_grad(P, dE0, dxds)
   # [P(i) \ dE0[i] - dot(dE0[i], dxds[i])*dxds[i] for i = 1:length(x)]
   F = forcing(precon, dE0⟂-Fk)

   res = maxres(P, dE0⟂)
   #maximum([norm(P(i)*dE0⟂[i],Inf) for i = 1:length(x)])

   return F, res

end
