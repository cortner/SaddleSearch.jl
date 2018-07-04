
function run!{T}(method::Union{ODEString, StaticString, LBFGSString}, E, dE, x0::Vector{T})
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
               (x_, P_, nit) -> forces(precon_scheme, x, x_, dE,
                                       direction(length(x), nit)),
                ref(x), log;
                g = (x_, P_) -> redistribute(x_, x, precon_scheme),
                tol = tol, maxnit=maxnit,
                method = "$(typeof(method))",
                verbose = verbose )

   x_return = verbose < 4 ? set_ref!(x, xout[end]) : [set_ref!(x, xout_n) for xout_n in xout]
   return x_return, log
end

function forces{T}(precon_scheme, x::Vector{T}, xref::Vector{Float64}, dE,
                  direction)
   @unpack precon, precon_prep! = precon_scheme

   x = set_ref!(x, xref)
   dxds = deepcopy(x)
   precon = precon_prep!(precon, x)
   Np = size(precon, 1)
   P(i) = precon[mod(i-1,Np)+1, 1]
   P(i, j) = precon[mod(i-1,Np)+1, mod(j-1,Np)+1]

   ds = [dist(precon_scheme, P, x, i) for i=1:length(x)-1]

   param = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   param /= param[end]; param[end] = 1.

   parametrise!(dxds, x, ds, parametrisation = param)

   dxds ./= point_norm(precon_scheme, P, dxds)
   dxds[1] = zeros(dxds[1]); dxds[end] = zeros(dxds[1])

   dE0_temp = [dE(x[i]) for i in direction]
   dE0 = [dE0_temp[i] for i in direction]

   dE0⟂ = proj_grad(precon_scheme, P, dE0, dxds)
   dE0⟂[1] = zeros(dE0⟂[1]); dE0⟂[end] = zeros(dE0⟂[1])
   F = forcing(precon_scheme, precon, dE0⟂)

   res = maxres(precon_scheme, P, dE0⟂)

   return F, res, length(param)
end
