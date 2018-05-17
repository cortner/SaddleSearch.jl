
using Dierckx
export ODENudgedElasticBandMethod

"""
`ODENudgedElasticbandMethod`: neb variant utilising adaptive time step ode solvers

### Parameters:
* `alpha` : step length
* 'k' : spring constrant
* 'abstol' : absolute errors tolerance
* 'reltol' : relative errors tolerance
* `tol` : residual tolerance
* `maxnit` : maximum number of iterations
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_cond` : true/false whether to precondition the minimisation step
"""
@with_kw type ODENudgedElasticBandMethod
   solver = ODE12r(rtol=1e-2)
   precon_scheme = localPrecon()
   path_traverse = serial()
   k::Float64
   # ------ shared parameters ------
   tol::Float64 = 1e-5
   maxnit::Int = 1000
   # precon = [I]
   # precon_prep! = (P, x) -> P
   verbose::Int = 2
   # precon_cond::Bool = false
end


function run!{T}(method::ODENudgedElasticBandMethod, E, dE, x0::Vector{T})
   # read all the parameters
   @unpack solver, precon_scheme, path_traverse, k, tol, maxnit,
            verbose = method
   @unpack direction = path_traverse
   # initialise variables
   x = copy(x0)
   nit = 0
   numdE, numE = 0, 0
   log = PathLog()
   # and just start looping
   if verbose >= 2
      @printf("SADDLESEARCH:  time | nit |  sup|∇E|_∞   \n")
      @printf("SADDLESEARCH: ------|-----|-----------------\n")
   end

   xout, log = odesolve(solver, (x_, P_, nit) -> forces(precon_scheme, x, x_, k, dE, direction(length(x), nit)),
   ref(x), log; tol = tol, maxnit=maxnit, method = "ODENEB", verbose = verbose)

   x = set_ref!(x, xout[end])
   return x, log
end

function forces{T}(precon_scheme, x::Vector{T}, xref::Vector{Float64},
                     k::Float64, dE, direction)
   @unpack precon, precon_prep! = precon_scheme
   x = set_ref!(x, xref)
   N = length(x)
   precon = precon_prep!(precon, x)
   Np = size(precon, 1); # P = i -> precon[mod(i-1,Np)+1]
   P(i) = precon[mod(i-1,Np)+1, 1]
   P(i, j) = precon[mod(i-1,Np)+1, mod(j-1,Np)+1]

   # central finite differences
   dxds = [[zeros(x[1])]; [0.5*(x[i+1]-x[i-1]) for i=2:N-1]; [zeros(x[1])]]
   dxds ./= point_norm(precon_scheme, P, dxds) # [norm(P(i), dxds[i]) for i=1:length(dxds)]
   # dxds = [ [zeros(dxds[1])]; dxds; [zeros(dxds[1])] ]
   d²xds² = [ [zeros(x[1])]; [x[i+1] - 2*x[i] + x[i-1] for i=2:N-1];
                                                         [zeros(x[1])] ]
   Fk = elastic_force(precon_scheme, P, k*N*N, dxds, d²xds²)
   #k*[dot(x[i+1] - 2*x[i] + x[i-1], P(i), dxds[i]) * dxds[i] for i=2:N-1]
   # Fk = [[zeros(x[1])]; Fk; [zeros(x[1])] ]

   # ord = M-mod(nit,2)*(M-1):2*mod(nit,2)-1:M-mod(nit+1,2)*(M-1)
   dE0_temp = [dE(x[i]) for i in direction]
   dE0 = [dE0_temp[i] for i in direction]

   dE0⟂ = proj_grad(precon_scheme, P, dE0, dxds)
   # [P(i) \ dE0[i] - dot(dE0[i], dxds[i])*dxds[i] for i = 1:length(x)]
   F = forcing(precon_scheme, precon, dE0⟂-Fk)

   res = maxres(precon_scheme, P, dE0⟂)
   #maximum([norm(P(i)*dE0⟂[i],Inf) for i = 1:length(x)])

   return F, res, N
end
