
using Dierckx

"""
`ODEStringMethod`: string variant utilising adaptive time step ode solvers.

### Parameters:
* `alpha` : step length
* 'abstol' : absolute errors tolerance
* 'reltol' : relative errors tolerance
* `tol_res` : residual tolerance
* `maxnit` : maximum number of iterations
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_cond` : true/false whether to precondition the minimisation step
"""

function run!{T}(method::ODEString, E, dE, x0::Vector{T})
   # read all the parameters
   @unpack tol, maxnit, precon_scheme, path_traverse, verbose = method
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
         (x_, P_, nit) -> forces(precon_scheme, x, x_, dE, direction(length(x), nit)),
         ref(x), log;
         g = (x_, P_) -> redistribute(x_, x, precon_scheme),
         tol = tol, maxnit=maxnit,
         method = "ODEString",
         verbose = verbose )

   x = set_ref!(x, xout[end])
   return x, log
end

function run!{T}(method::StaticString, E, dE, x0::Vector{T})
   # read all the parameters
   @unpack tol, maxnit, precon_scheme, path_traverse,
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

   xout, log = odesolve(solver(method),
         (x_, P_, nit) -> forces(precon_scheme, x, x_, dE, direction(length(x), nit)),
         ref(x), log;
         g = (x_, P_) -> redistribute(x_, x, precon_scheme),
         tol = tol, maxnit=maxnit,
         method = "StaticString",
         verbose = verbose )

   x = set_ref!(x, xout[end])
   return x, log
end

function forces{T}(precon_scheme, x::Vector{T}, xref::Vector{Float64}, dE, direction)
   @unpack precon, precon_prep! = precon_scheme

   x = set_ref!(x, xref)
   t = copy(x)
   precon = precon_prep!(precon, x)
   Np = size(precon, 1)
   P(i) = precon[mod(i-1,Np)+1, 1]
   P(i, j) = precon[mod(i-1,Np)+1, mod(j-1,Np)+1]

   ds = [dist(precon_scheme, P, x, i) for i=1:length(x)-1]

   param = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   param /= param[end]; param[end] = 1.

   parametrise!(t, x, ds, parametrisation = param)

   t ./= point_norm(precon_scheme, P, t)
   t[1] =zeros(t[1]); t[end]=zeros(t[1])

   dE0_temp = [dE(x[i]) for i in direction]
   dE0 = [dE0_temp[i] for i in direction]

   dE0⟂ = proj_grad(precon_scheme, P, dE0, t)
   F = forcing(precon_scheme, precon, dE0⟂)

   res = maxres(precon_scheme, P, dE0⟂)

   return F, res, length(param)
end
