
using Dierckx
export ODEStringMethod

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
@with_kw type ODEStringMethod
   solver = ODE12r(rtol=1e-2)
   precon_scheme = localPrecon()
   path_traverse = serial()
   # ------ shared parameters ------
   tol_res::Float64 = 1e-5
   maxnit::Int = 1000
   verbose::Int = 2
end

function run!{T}(method::ODEStringMethod, E, dE, x0::Vector{T})
   # read all the parameters
   @unpack solver, precon_scheme, path_traverse, tol_res, maxnit, verbose = method
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

   xout, log = odesolve(solver,
         (x_, P_, nit) -> forces(precon_scheme, x, x_, dE, direction(length(x), nit)),
         ref(x), log;
         g = (x_, P_) -> redistribute(x_, x, precon_scheme),
         tol_res = tol_res, maxnit=maxnit,
         method = "ODEStringMethod",
         verbose = verbose )

   x = set_ref!(x, xout[end])
   return x, log
end

function forces{T}(precon_scheme, x::Vector{T}, xref::Vector{Float64}, dE, direction)
   @unpack precon, precon_prep!, precon_cond, dist, point_norm,
            proj_grad, forcing, maxres = precon_scheme

   x = set_ref!(x, xref)
   t = copy(x)
   precon = precon_prep!(precon, x)
   Np = size(precon, 1);
   function P(i) return precon[mod(i-1,Np)+1, 1]; end
   function P(i, j) return precon[mod(i-1,Np)+1, mod(j-1,Np)+1]; end

   ds = [dist(P, x, i) for i=1:length(x)-1]

   param = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   param /= param[end]; param[end] = 1.

   parametrise!(x, t, ds, parametrisation = param)

   t ./= point_norm(P, t)
   t[1] =zeros(t[1]); t[end]=zeros(t[1])

   dE0_temp = [dE(x[i]) for i in direction]
   dE0 = [dE0_temp[i] for i in direction]

   dE0⟂ = proj_grad(P, dE0, t)
   F = forcing(precon, dE0⟂)

   res = maxres(P, dE0⟂)

   return F, res, length(param)
end
