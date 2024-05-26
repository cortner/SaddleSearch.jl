
function run!(method::Union{ODENEB, StaticNEB}, E, dE, x0::Path{T,NI}) where {T,NI}
   # read all the parameters
   @unpack k, interp, tol, maxtol, maxnit, precon_scheme, path_traverse, fixed_ends,
            verbose = method
   @unpack precon, precon_prep! = precon_scheme
   @unpack direction = path_traverse
   # initialise variables
   x = x0.x

   nit = 0
   numdE, numE = 0, 0
   log = PathLog()
   # and just start looping
   if verbose >= 2
       @printf("SADDLESEARCH:         k  =  %1.2e        <- parameters\n", k)
   end
   file = []
   if verbose >= 4
       dt = Dates.format(now(), "d-m-yyyy_HH:MM")
       file = open("log_$(dt).txt", "w")
       strlog = @sprintf("SADDLESEARCH:         k  =  %1.2e        <- parameters\n", k)
       write(file, strlog)
       flush(file)
   end
   xout, log, alpha = odesolve(solver(method),
               (X, P, nit) -> forces(P, typeof(x0), X, dE, precon_scheme,
                                       direction(NI, nit), k, interp, fixed_ends),
               vec(x), log; file = file,
               tol = tol, maxtol = maxtol, maxnit = maxnit,
               P = precon,
               precon_prep! = (P, X) -> precon_prep!(P, convert(typeof(x0), X)),
               method = "$(typeof(method))",
               verbose = verbose)

   x_return = verbose < 4 ? convert(typeof(x0), xout[end]) : [convert(typeof(x0), xout_n) for xout_n in xout]
   return x_return, log, alpha
end

function run!(method::AccelNEB, E, dE, ddE, x0::Path{T,NI}) where {T,NI}
   # read all the parameters
   @unpack k, interp, tol, maxtol, maxnit, precon_scheme, path_traverse, fixed_ends,
            verbose = method
   @unpack precon, precon_prep! = precon_scheme
   @unpack direction = path_traverse
   # initialise variables
   x = x0.x

   nit = 0
   numdE, numE = 0, 0
   log = PathLog()
   # and just start looping
   if verbose >= 2
       @printf("SADDLESEARCH:         k  =  %1.2e        <- parameters\n", k)
   end
   file = []
   if verbose >= 4
       dt = Dates.format(now(), "d-m-yyyy_HH:MM")
       file = open("log_$(dt).txt", "w")
       strlog = @sprintf("SADDLESEARCH:         k  =  %1.2e        <- parameters\n", k)
       write(file, strlog)
       flush(file)
   end

   xout, log, alpha = odesolve(solver(method),
               (X, P, nit) -> forces(P, typeof(x0), X, dE, precon_scheme,
                                       direction(NI, nit), k, interp, fixed_ends),
               (X, P) -> jacobian(P, typeof(x0), X, dE, ddE, k),
               vec(x), log; file = file,
               tol = tol, maxtol=maxtol, maxnit=maxnit,
               P = precon,
               precon_prep! = (P, X) -> precon_prep!(P, convert(typeof(x0), X)),
               method = "$(typeof(method))",
               verbose = verbose)

   x_return = verbose < 4 ? convert(typeof(x0), xout[end]) : [convert(typeof(x0), xout_n) for xout_n in xout]
   return x_return, log, alpha
end

# forcing term for NEB method
function forces(precon, path_type::Type{Path{T,NI}}, X::Vector{Float64}, dE, precon_scheme,
                  direction, k::Float64, interp::Int, fixed_ends::Bool) where {T,NI}

   x = convert(path_type, X)
   dxds = deepcopy(x)

   # preconditioner
   Np = size(precon, 1); N = length(x)
   P(i) = precon[mod(i-1,Np)+1, 1]
   P(i, j) = precon[mod(i-1,Np)+1, mod(j-1,Np)+1]

   # interpolate path to find tangents and 2nd derivatives
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

   # elastic interactions between adjacent images
   Fk = elastic_force(precon_scheme, P, k*N*N, dxds, d²xds²)

   # potential gradient
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

   # projecting out tangent term of potential gradient
   dE0perp = proj_grad(precon_scheme, P, dE0, dxds)

   # collecting force term
   F = forcing(precon_scheme, precon, dE0perp-Fk)

   # residual error
   res = maxres(precon_scheme, P, dE0perp)

   return F, res, cost, (X, Y) -> dot_P(precon_scheme, convert(path_type, X), P, convert(path_type, Y))
end

function jacobian(precon, path_type::Type{Path{T,NI}}, X::Vector{Float64},
   dE, ddE, k::Float64) where {T,NI}
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
   [J[n,n-1] = ∂Fⁿ⁻(x, n, dE, P) + ∂Sⁿ⁻(k, x, n, P) for n=2:N-1]
   if Np==1 && P(1)==I
      [J[n,n] = δFⁿ(x, n, H, P, H) + ∂Sⁿ(k, x, n, P) for n=1:N]
   else
      [J[n,n] = δFⁿ(x, n, H, P, n -> I) + ∂Sⁿ(k, x, n, P) for n=1:N]
   end
   [J[n,n+1] = ∂Fⁿ⁺(x, n, dE, P) + ∂Sⁿ⁺(k, x, n, P) for n=2:N-1]

   return ref(J)
end
