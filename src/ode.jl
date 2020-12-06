
"""
`Euler`: simple Euler method ODE solver with fixed step length.

### Parameters:
* `h` : step length
"""
@with_kw struct Euler
   h::Float64 = 1e-1
end

function odesolve(solver::Euler, f, X0::Vector{Float64}, log::IterationLog;
                  file = nothing,
                  verbose = 1,
                  g=(X, P)->X, tol=1e-4, maxtol=1e3, maxnit=100,
                  P = I, precon_prep! = (P, X) -> P,
                  method = "Static" )

   @unpack h = solver
   log_header(verbose, file, :h => h)

   X = copy(X0)
   P = precon_prep!(P, X)

   Xout = [];

   numdE, numE = 0, 0

   # initialise variables
   X = g(X, P)
   P = precon_prep!(P, X)
   Fn, Rn, ndE, _ = f(X, P, 0)
   numdE += ndE

   push!(Xout, X) # store X
   push!(log, numE, numdE, Rn) # residual, store history

   log_history(verbose, file, 0, Rn)
   if Rn <= tol
      log_converged(verbose, file, method, 0)
      return Xout, log, h
   end

   for nit = 1:maxnit
      # redistribute
      Xnew = g(X + h * Fn, P)

      # return force
      Pnew = precon_prep!(P, Xnew)
      Fnew, Rnew, ndE, _ = f(Xnew, Pnew, nit)

      numdE += ndE

      X, Fn, Rn, P = Xnew, Fnew, Rnew, Pnew
      push!(Xout, X) # store X
      push!(log, numE, numdE, Rn) # residual, store history

      log_history(verbose, file, nit, Rn)
      if Rn <= tol
         log_converged(verbose, file, method, nit)
         return Xout, log, h
      end
   end

   # logging
   log_diverged(verbose, file, method, maxnit)
   return Xout, log, h
end


"""
`ODE12r`: adaptive ODE solver, uses 1st and 2nd order approximations to estimate local error and find a new step length

### Parameters:
* `rtol` : relative tolerance
* `threshold` : threshold for error estimate
* `C1` : sufficient contraction parameter
* `C2` : residual growth control (Inf means there is no control)
* `h` : step size, if nothing is passed, an estimate is used based on ODE12
* `hmin` : minimal allowed step size
* `maxF` : terminate if |Fn| > maxF * |F0|
* `extrapolate` : extrapolation style (3 seems the most robust)
"""


@with_kw struct ODE12r
   rtol::Float64 = 1e-1
   threshold::Float64 = 1.0
   C1::Float64 = 1e-2
   C2::Float64 = 2.0
   h = nothing
   hmin::Float64 = 1e-10
   maxF::Float64 = 1e3
   extrapolate::Int = 3
end

function odesolve(solver::ODE12r, f, X0::Vector{Float64}, log::IterationLog;
                  file = nothing,
                  verbose = 1,
                  g=(X, P)->X, tol=1e-4, maxtol=1e3, maxnit=100,
                  P = I, precon_prep! = (P, X) -> P,
                  method = "ODE" )

   @unpack threshold, rtol, C1, C2, h, hmin, extrapolate = solver
   log_header(verbose, file, :rtol => rtol, :threshold => threshold)

   X = copy(X0)
   P = precon_prep!(P, X)

   Xout = [];

   numdE, numE = 0, 0

   # initialise variables
   X = g(X, P)
   P = precon_prep!(P, X)
   Fn, Rn, ndE, _ = f(X, P, 0)
   numdE += ndE

   push!(Xout, X)
   push!(log, numE, numdE, Rn)

   # logging
   log_history(verbose, file, 0, Rn)
   if Rn <= tol
      log_converged(verbose, file, method, 0)
      return Xout, log, h
   end

   if Rn >= maxtol
      warn_maxtol(verbose, file, 0, Rn)
      push!(Xout, X) # store X
      push!(log, typemax(Int64), typemax(Int64), Rn) # residual, store history
      return Xout, log, h
   end

   # computation of the initial step
   r = norm(Fn ./ max.(abs.(X), threshold), Inf) + floatmin(Float64)
   if h == nothing
      h = 0.5 * rtol^(1/2) / r
      h = max(h, hmin)
   end

   for nit = 1:maxnit
      # redistribute
      Xnew = g(X + h * Fn, P)   # that way it implicitly becomes part
                                # of `f`
                                # but it seems to make the evolution slower; need more testing!
      # return force
      Pnew = precon_prep!(P, Xnew)
      Fnew, Rnew, ndE, dot_P = f(Xnew, Pnew, nit)

      numdE += ndE

      # error estimation
      e = 0.5 * h * (Fnew - Fn)
      err = norm(e ./ max.(maximum([abs.(X) abs.(Xnew)],dims=2), threshold), Inf) + floatmin(Float64)

      # accept step if residual is sufficient decreased
      if (   ( Rnew <= Rn * (1 - C1 * h) )         # contraction
          || ( Rnew <= Rn * C2 && err <= rtol ) )  # moderate growth + error control
         accept = true
      else
         accept = false
         conditions = (Rnew <= Rn * (1 - C1 * h), Rnew <= Rn * C2, err <= rtol )
      end

      # whether we accept or reject this step, we now need a good guess for
      # the next step-size, from a line-search-like construction
      y = Fn - Fnew
      if extrapolate == 1       # F(xn + h Fn) ⋅ Fn ~ 0
         h_ls = h * dot_P(Fn, Fn) / dot_P(Fn, y)
      elseif extrapolate == 2   # F(Xn + h Fn) ⋅ F{n+1} ~ 0
         h_ls = h * dot_P(Fn, Fnew) / (dot_P(Fn, y) + 1e-10)
      elseif extrapolate == 3   # min | F(Xn + h Fn) |
         h_ls = h * dot_P(Fn, y) / (dot_P(y, y) + 1e-10)
      else
         error_parameter(verbose, file, "`extrapolate`")
      end
      if isnan(h_ls) || (h_ls < hmin)
         h_ls = Inf
      end
      # or from the error estimate
      h_err = h * 0.5 * sqrt(rtol/err)

      if accept
         X, Fn, Rn, P  = Xnew, Fnew, Rnew, Pnew

         push!(Xout, X) # store X
         push!(log, numE, numdE, Rn) # residual, store history

         # logging
         log_history(verbose, file, nit, Rn)
         if Rn <= tol
            log_converged(verbose, file, method, nit)
            return Xout, log, h
         end

         if Rn >= maxtol
            warn_maxtol(verbose, file, nit, Rn)
            push!(Xout, X) # store X
            push!(log, typemax(Int64), typemax(Int64), Rn) # residual, store history
            return Xout, log, h
         end

         # Compute a new step size.
         h = max(0.25 * h, min(4*h, h_err, h_ls))
         # log step-size analytic results
         log_acceptstep(verbose, file, h, h_ls, h_err, Rn)
      else
         # compute new step size
         h = max(0.1 * h, min(0.25 * h, h_err, h_ls))
         # log step-size analytic results
         log_rejectstep(verbose, file, h, Rnew, Rn)
      end

      # error message if step size is too small
      if abs(h) <= hmin
         warn_hmin(verbose, file, h, nit)
         return Xout, log, h
      end
   end

   # logging
   log_diverged(verbose, file, method, maxnit)
   return Xout, log, h
end

@with_kw struct momentum_descent
   h::Float64 = 1e-1
   b = 1e-1
   fd_scheme = :central
   redistrib = :canonical
   # adaptive first step
   rtol::Float64 = 1e-1
   threshold::Float64 = 1.0
   C1::Float64 = 1e-2
   C2::Float64 = 2.0
   h0 = nothing
   hmin::Float64 = 1e-10
   maxF::Float64 = 1e3
   extrapolate::Int = 3
end

function odesolve(solver::momentum_descent, f, df, X0::Vector{Float64},
                  log::IterationLog;
                  file = nothing,
                  verbose = 1,
                  g=(X, P)->X, tol=1e-4, maxtol=1e3, maxnit=100,
                  P = I, precon_prep! = (P, X) -> P,
                  method = "Momentum Descent" )

   # @unpack h, b, finite_diff = solver
   @unpack h, b, fd_scheme, redistrib,
            threshold, rtol, C1, C2, h0, hmin, extrapolate = solver

   log_header(verbose, file, :h => h)

   X = copy(X0)
   P = precon_prep!(P, X)

   Xout = [];

   numdE, numE = 0, 0
   # initialise variables
   X = g(X, P)
   P = precon_prep!(P, X)
   Fn, Rn, ndE, _ = f(X, P, 0)
   numdE += ndE

   push!(Xout, X) # store X
   push!(log, numE, numdE, Rn) # residual, store history

   # logging
   log_history(verbose, file, 0, Rn)
   if Rn <= tol
      log_converged(verbose, file, method, 0)
      return Xout, log, h
   end

   r = norm(Fn ./ max.(abs.(X), threshold), Inf) + floatmin(Float64)
   if h0 == nothing
      h0 = 0.5 * rtol^(1/2) / r
      h0 = max(h0, hmin)
   end

   Xnew = g(X + h0 * Fn, P)

   # return force
   Pnew = precon_prep!(P, Xnew)
   Fnew, Rnew, ndE, _ = f(Xnew, Pnew, 1)

   numdE += ndE

   X, Fn, Rn, P = Xnew, Fnew, Rnew, Pnew
   push!(Xout, X) # store X
   push!(log, numE, numdE, Rn) # residual, store history

   # logging
   log_history(verbose, file, 1, Rn)
   if Rn <= tol
      log_converged(verbose, file, method, 1)
      return Xout, log, h
   end

   if b == nothing
      dFn = -df(X, P)
      Λ = eigen(Matrix(dFn)).values
      λmax  = Λ[findmax(real(Λ))[2]]
      _, b = stability(λmax)
      b = b*.5

      h = 1.0; it = 1; it_max = 100
      while (it<=it_max && !minimum([criterion(fd_scheme, λ*h*h, b*h) for λ in Λ[real(Λ).>0.]]))
         h = h/2
         it+=1
      end
   end

   log_header(verbose, file, :b => b, :h => h)

   for nit = 2:maxnit
      # if b == nothing
      #    if mod(nit, 50) == 0
      #       dFn = -df(X, P)
      #       Λ = eigen(dFn).values
      #       λmax  = Λ[findmax(real(Λ))[2]]
      #       _, b = stability(λmax)
      #       b = b*.5
      #
      #       h = 1.0; it = 1; it_max = 100
      #       while (it<=it_max && !minimum([criterion(fd_scheme, λ*h*h, b*h) for λ in Λ[real(Λ).>0.]]))
      #           h = h/2
      #           it+=1
      #       end
      #       @printf("b = %1.2e, h = %1.2e\n",b, h)
      #    end
      # end
      # redistribute
      if redistrib == :canonical
         Xnew = g(finite_diff(fd_scheme, Xout, Fn, b, h, true), P)
      elseif redristib == :dynamic
         Xnew = finite_diff(fd_scheme, Xout, g(X + h * Fn, P), b, h, false)
      end

      # return force
      Pnew = precon_prep!(P, Xnew)
      Fnew, Rnew, ndE, _ = f(Xnew, Pnew, nit)

      numdE += ndE

      X, Fn, Rn, P = Xnew, Fnew, Rnew, Pnew
      push!(Xout, X) # store X
      push!(log, numE, numdE, Rn) # residual, store history

      # logging
      log_history(verbose, file, nit, Rn)
      if Rn <= tol
         log_converged(verbose, file, method, nit)
         return Xout, log, h
      end

      if Rn >= maxtol
         warn_maxtol(verbose, file, nit, Rn)
         push!(Xout, X) # store X
         push!(log, typemax(Int64), typemax(Int64), Rn) # residual, store history
         return Xout, log, h
      end

   end

   # logging
   log_diverged(verbose, file, method, maxnit)

   if verbose >= 4 && file!=nothing
      close(file)
   end
   return Xout, log, h
end

@with_kw struct ODEmomentum_descent
   h = nothing
   b = 1e-1
   fd_scheme = :central
   redistrib = :canonical
   # adaptive first step
   rtol::Float64 = 1e-1
   threshold::Float64 = 1.0
   C1::Float64 = 1e-2
   C2::Float64 = 2.0
   h0 = nothing
   hmin::Float64 = 1e-10
   maxF::Float64 = 1e3
   extrapolate::Int = 3
end

function odesolve(solver::ODEmomentum_descent, f, df, X0::Vector{Float64},
                  log::IterationLog;
                  file = nothing,
                  verbose = 1,
                  g=(X, P)->X, tol=1e-4, maxtol=1e3, maxnit=100,
                  P = I, precon_prep! = (P, X) -> P,
                  method = "Adaptive Momentum Descent", path_type = typeof(X0))

   @unpack h, b, fd_scheme, redistrib,
            threshold, rtol, C1, C2, h0, hmin, extrapolate = solver

   log_header(verbose, file, :h => h)

   X = copy(X0)
   P = precon_prep!(P, X)

   Xout = []; Vout =[];

   numdE, numE = 0, 0
   # initialise variables
   X = g(X, P)
   P = precon_prep!(P, X)
   Fn, Rn, ndE, _ = f(X, P, 0)
   numdE += ndE

   push!(Xout, X) # store X
   push!(Vout, -Inf) # pseudo value
   push!(log, numE, numdE, Rn) # residual, store history

   # logging
   log_history(verbose, file, 0, Rn)
   if Rn <= tol
      log_converged(verbose, file, method, 0)
      return Xout, Vout, log, -Inf
   end

   rtol0 = 1e-2
   r = norm(Fn ./ max.(abs.(X), threshold), Inf) + floatmin(Float64)
   if h0 == nothing
      h0 = 0.5 * rtol0^(1/2) / r
      h0 = max(h0, hmin)
   end

   Xnew = g(X + h0 * Fn, P)

   # return force
   Pnew = precon_prep!(P, Xnew)
   Fnew, Rnew, ndE, _ = f(Xnew, Pnew, 1)

   numdE += ndE

# TODO: ? Vnew = Fn ? where is g applied ? Think about it
   Vnew  = (Xnew - X)/h0
   X, V, Fn, Rn, P = Xnew, Vnew, Fnew, Rnew, Pnew
   push!(Xout, X) # store X
   push!(log, numE, numdE, Rn) # residual, store history
   push!(Vout, V)

   # logging

   log_history(verbose, file, 1, Rn)
   if Rn <= tol
      log_converged(verbose, file, method, 1)
      return Xout, Vout, log, h0
   end

   if b == nothing
      dFn = -df(X, P)
      Λ = eigen(Matrix(dFn)).values
      λmax  = Λ[findmax(real(Λ))[2]]
      _, b = stability(λmax)
      b = b*.5

      h = 1.0; it = 1; it_max = 100
      while (it<=it_max && !minimum([criterion(fd_scheme, λ*h*h, b*h) for λ in Λ[real(Λ).>0.]]))
         h = h/2
         it+=1
      end
   end

   log_header(verbose, file, :b => b)

   U = [X; V]
   @show(norm(U, Inf), norm(X, Inf), norm(V, Inf))
   Ix = 1:length(X)
   Iv = length(X)+1:2*length(X)

   Log = PathLog()

   adaptive_solver = ODE12r(rtol=rtol, threshold=threshold, h=h)
   Uout, Log, Alpha = odesolve(adaptive_solver,
               (U, P, nit) -> central_diff_forces(U[Ix], U[Iv], path_type, P, f, b, nit),
               U, Log; file = file,
               tol = tol, maxtol = maxtol, maxnit = maxnit,
               P = P,
               precon_prep! = (P, U) -> [I],
               method = "$(typeof(method))",
               verbose = verbose)

   # logging
   # log_diverged(verbose, file, method, maxnit)
   if verbose >= 4 && file!=nothing
      close(file)
   end
   return Xout, Vout, log, h
end
