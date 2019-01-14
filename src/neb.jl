
# function run!{T}(method::Union{ODENEB, StaticNEB}, E, dE, x0::Vector{T})
function run!{T,NI}(method::Union{ODENEB, StaticNEB}, E, dE, x0::Path{T,NI})
   # read all the parameters
   @unpack k, interp, tol, maxnit, precon_scheme, path_traverse, fixed_ends,
            verbose = method
   @unpack precon, precon_prep! = precon_scheme
   @unpack direction = path_traverse
   # initialise variables
   x = x0.x
   # x = copy(x0)
   # xref = deepcopy(x0)
   nit = 0
   numdE, numE = 0, 0
   log = PathLog()
   # and just start looping
   if verbose >= 2
       @printf("SADDLESEARCH:         k  =  %1.2e        <- parameters\n", k)
   end

   if verbose >= 4
       dt = Dates.format(now(), "d-m-yyyy_HH:MM")
       file = open("log_$(dt).txt", "w")
       strlog = @sprintf("SADDLESEARCH:         k  =  %1.2e        <- parameters\n", k)
       write(file, strlog)
       flush(file)
   end
   xout, log = odesolve(solver(method),
               (X, P, nit) -> forces(P, typeof(x0), X, dE, precon_scheme,
                                       direction(NI, nit), k, interp, fixed_ends),
               vec(x), log, file;
               tol = tol, maxnit=maxnit,
               P = precon,
               precon_prep! = (P, X) -> precon_prep!(P, convert(typeof(x0), X)),
               method = "$(typeof(method))",
               verbose = verbose)

   x_return = verbose < 4 ? convert(typeof(x0), xout[end]) : [convert(typeof(x0), xout_n) for xout_n in xout]
   return x_return, log
end

function forces{T,NI}(precon, path_type::Type{Path{T,NI}}, X::Vector{Float64}, dE, precon_scheme,
                  direction, k::Float64, interp::Int, fixed_ends::Bool)

   x = convert(path_type, X)
   dxds = deepcopy(x)

   Np = size(precon, 1); N = length(x)
   P(i) = precon[mod(i-1,Np)+1, 1]
   P(i, j) = precon[mod(i-1,Np)+1, mod(j-1,Np)+1]

   if interp == 1
      # central finite differences
      dxds = [[zeros(x[1])]; [0.5*(x[i+1]-x[i-1]) for i=2:N-1]; [zeros(x[1])]]
      d²xds² = [[zeros(x[1])]; [x[i+1] - 2*x[i] + x[i-1] for i=2:N-1];
               [zeros(x[1])]]
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
   dxds[1] = zeros(dxds[1]); dxds[end] = zeros(dxds[1])

   Fk = elastic_force(precon_scheme, P, k*N*N, dxds, d²xds²)

   dE0_temp = []
   if !fixed_ends
      dE0_temp = [dE(x[i]) for i in direction]
      cost = N
   else
      dE0_temp = [[zeros(x[1])]; [dE(x[i]) for i in direction[2:end-1]];
                  [zeros(x[1])]]
      cost = N - 2
   end
   dE0 = [dE0_temp[i] for i in direction]

   dE0⟂ = proj_grad(precon_scheme, P, dE0, dxds)
   F = forcing(precon_scheme, precon, dE0⟂-Fk)

   res = maxres(precon_scheme, P, dE0⟂)

   return F, res, cost, (X, Y) -> dot_P(precon_scheme, convert(path_type, X), P, convert(path_type, Y))
end
