
using Dierckx
# export ODENudgedElasticBandMethod

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
function run!{T}(method::ODENEB, E, dE, x0::Vector{T})
   # read all the parameters
   @unpack k, interp, tol, maxnit, precon_scheme, path_traverse, verbose = method
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

   xout, log = odesolve(solver(method),
   (x_, P_, nit) -> forces(precon_scheme, x, x_, dE, direction(length(x), nit), k, interp),
   ref(x), log;
   tol = tol, maxnit=maxnit,
   method = "ODENEB",
   verbose = verbose)

   x = set_ref!(x, xout[end])
   return x, log
end

function run!{T}(method::StaticNEB, E, dE, x0::Vector{T})
   # read all the parameters
   @unpack k, interp, tol, maxnit, precon_scheme, path_traverse, verbose = method
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

   xout, log = odesolve(solver(method),
   (x_, P_, nit) -> forces(precon_scheme, x, x_, dE, direction(length(x), nit), k, interp),
   ref(x), log;
   tol = tol, maxnit=maxnit,
   method = "StaticNEB",
   verbose = verbose)

   x = set_ref!(x, xout[end])
   return x, log
end

function forces{T}(precon_scheme, x::Vector{T}, xref::Vector{Float64}, dE, direction,
                     k::Float64, interp::Int)
   @unpack precon, precon_prep! = precon_scheme
   x = set_ref!(x, xref)
   dxds = copy(x)
   precon = precon_prep!(precon, x)
   Np = size(precon, 1); N = length(x)
   P(i) = precon[mod(i-1,Np)+1, 1]
   P(i, j) = precon[mod(i-1,Np)+1, mod(j-1,Np)+1]

   # central finite differences
   if interp == 1
       # central finite differences
       dxds = [[zeros(x[1])]; [0.5*(x[i+1]-x[i-1]) for i=2:N-1]; [zeros(x[1])]]
       d²xds² = [[zeros(x[1])]; [x[i+1] - 2*x[i] + x[i-1] for i=2:N-1]; [zeros(x[1])]]
   elseif interp > 1
       # splines
       ds = [dist(precon_scheme, P, x, i) for i=1:length(x)-1]

       param = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
       param /= param[end]; param[end] = 1.

       d²xds² = parametrise!(dxds, x, ds, parametrisation = param)
       k *= (1/(N*N))
   else
       error("SADDLESEARCH: invalid `interpolate` parameter")
   end

   dxds ./= point_norm(precon_scheme, P, dxds)
   dxds[1]=zeros(dxds[1]); dxds[end]=zeros(dxds[1])

   Fk = elastic_force(precon_scheme, P, k*N*N, dxds, d²xds²)

   dE0_temp = [dE(x[i]) for i in direction]
   dE0 = [dE0_temp[i] for i in direction]

   dE0⟂ = proj_grad(precon_scheme, P, dE0, dxds)
   F = forcing(precon_scheme, precon, dE0⟂-Fk)

   res = maxres(precon_scheme, P, dE0⟂)

   return F, res, N
end
