# using SaddleSearch: run!


function run!(method::Union{ODEString, StaticString}, E, dE, x0::Path{T,NI}) where {T, NI}
   # read all the parameters
   @unpack tol, maxtol, maxnit, precon_scheme, path_traverse, fixed_ends, verbose = method
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
               tol = tol, maxtol = maxtol, maxnit=maxnit,
               P = precon,
               precon_prep! = (P, X) -> precon_prep!(P, convert(typeof(x0), X)),
               method = "$(typeof(method))",
               verbose = verbose )

   x_return = verbose < 4 ? convert(typeof(x0), xout[end]) : [convert(typeof(x0), xout_n) for xout_n in xout]
   return x_return, log, alpha
end

function run!(method::AccelString, E, dE, ddE, x0::Path{T,NI}) where {T, NI}
   # read all the parameters
   @unpack tol, maxtol, maxnit, precon_scheme, path_traverse, fixed_ends, verbose = method
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
               (X, P) -> jacobian(P, typeof(x0), X, dE, ddE),
               vec(x), log; file = file,
               g = (X, P) -> redistribute(X, typeof(x0), P, precon_scheme),
               tol = tol, maxtol = maxtol, maxnit = maxnit,
               P = precon,
               precon_prep! = (P, X) -> precon_prep!(P, convert(typeof(x0), X)),
               method = "$(typeof(method))",
               verbose = verbose )

   x_return = verbose < 4 ? convert(typeof(x0), xout[end]) : [convert(typeof(x0), xout_n) for xout_n in xout]
   return x_return, log, alpha
end

# forcing term for string method
function forces(precon, path_type::Type{Path{T,NI}}, X::Vector{Float64}, dE,
                  precon_scheme, direction, fixed_ends::Bool) where {T, NI}

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
   dxds[1] = zeros(size(dxds[1])); dxds[end] = zeros(size(dxds[1]))

   # potential gradient
   dE0_temp = []
   if !fixed_ends
      dE0_temp = [dE(x[i]) for i in direction]
      cost = length(param)
   else
      dE0_temp = [[zeros(size(x[1]))]; [dE(x[i]) for i in direction[2:end-1]];
                  [zeros(size(x[1]))]]
      cost = length(param) - 2
   end
   dE0 = [dE0_temp[i] for i in direction]

   # projecting out tangent term of potential gradient
   dE0perp = proj_grad(precon_scheme, P, dE0, dxds)

   # collecting force term
   F = forcing(precon_scheme, precon, dE0perp)

   # residual error
   res = maxres(precon_scheme, P, dE0perp)

   return F, res, cost, (X, Y) -> dot_P(precon_scheme, convert(path_type, X), P, convert(path_type, Y))
end

function jacobian(precon, path_type::Type{Path{T,NI}}, X::Vector{Float64},
                    dE, ddE) where {T, NI}

   x = convert(path_type, X)

   # preconditioner
   Np = size(precon, 1)
   P(i) = precon[mod(i-1,Np)+1, 1]
   P(i, j) = precon[mod(i-1,Np)+1, mod(j-1,Np)+1]

   hessian = ddE.precon; hessian_prep! = ddE.precon_prep!
   hessian = hessian_prep!(hessian, x)
   H(i) = hessian[mod(i-1,Np)+1, 1]
   H(i, j) = hessian[mod(i-1,Np)+1, mod(j-1,Np)+1]

   N = length(x); M = length(x[1])
   O = zeros(M, M); J = fill(O,(N, N))
   [J[n,n-1] = ∂Fⁿ⁻(x, n, dE, P) for n=2:N-1]
   if Np==1 && P(1)==I
      [J[n,n] = δFⁿ(x, n, H, P, H) for n=1:N]
   else
      [J[n,n] = δFⁿ(x, n, H, P, n -> I) for n=1:N]
   end
   [J[n,n+1] = ∂Fⁿ⁺(x, n, dE, P) for n=2:N-1]

   return ref(J)
end
