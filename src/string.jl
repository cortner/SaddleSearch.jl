
function run!{T,NI}(method::Union{ODEString, StaticString, AccelString}, E, dE, x0::Path{T,NI})
   # read all the parameters
   @unpack tol, maxnit, precon_scheme, path_traverse, fixed_ends, verbose = method
   @unpack precon, precon_prep! = precon_scheme
   @unpack direction = path_traverse
   # initialise variables
   x = x0.x

   nit = 0
   numdE, numE = 0, 0
   log = PathLog()
   # and just start looping
   file =[]
   if verbose >= 4
       dt = Dates.format(now(), "d-m-yyyy_HH:MM")
       file = open("log_$(dt).txt", "w")
   end
   xout, log, alpha = odesolve(solver(method),
               (X, P, nit) -> forces(P, typeof(x0), X, dE, precon_scheme,
                                       direction(NI, nit), fixed_ends),
                vec(x), log; file = file,
                g = (X, P) -> redistribute(X, typeof(x0), P, precon_scheme),
                tol = tol, maxnit=maxnit,
                P = precon,
                precon_prep! = (P, X) -> precon_prep!(P, convert(typeof(x0), X)),
                method = "$(typeof(method))",
                verbose = verbose )

   x_return = verbose < 4 ? convert(typeof(x0), xout[end]) : [convert(typeof(x0), xout_n) for xout_n in xout]
   return x_return, log, alpha
end

# forcing term for string method
function forces{T,NI}(precon, path_type::Type{Path{T,NI}}, X::Vector{Float64}, dE,
                  precon_scheme, direction, fixed_ends::Bool)

   x = convert(path_type, X)
   dxds = deepcopy(x)

   # preconditioner
   Np = size(precon, 1)
   P(i) = precon[mod(i-1,Np)+1, 1]
   P(i, j) = precon[mod(i-1,Np)+1, mod(j-1,Np)+1]

   # find tangents
   ds = [dist(precon_scheme, P, x, i) for i=1:length(x)-1]
   param = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   param /= param[end]; param[end] = 1.
   parametrise!(dxds, x, ds, parametrisation = param)
   dxds ./= point_norm(precon_scheme, P, dxds)
   dxds[1] = zeros(dxds[1]); dxds[end] = zeros(dxds[1])

   # potential gradient
   dE0_temp = []
   if !fixed_ends
      dE0_temp = [dE(x[i]) for i in direction]
      cost = length(param)
   else
      dE0_temp = [[zeros(x[1])]; [dE(x[i]) for i in direction[2:end-1]];
                  [zeros(x[1])]]
      cost = length(param) - 2
   end
   dE0 = [dE0_temp[i] for i in direction]

   # projecting out tangent term of potential gradient
   dE0⟂ = proj_grad(precon_scheme, P, dE0, dxds)

   # collecting force term
   F = forcing(precon_scheme, precon, dE0⟂)

   # residual error
   res = maxres(precon_scheme, P, dE0⟂)

   return F, res, cost, (X, Y) -> dot_P(precon_scheme, convert(path_type, X), P, convert(path_type, Y))
end
