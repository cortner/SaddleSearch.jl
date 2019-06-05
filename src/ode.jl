
"""
`Euler`: simple Euler method ODE solver with fixed step length.

### Parameters:
* `h` : step length
"""
@with_kw type Euler
   h::Float64 = 1e-1
end

function odesolve(solver::Euler, f, X0::Vector{Float64}, log::IterationLog;
                  file = nothing,
                  verbose = 1,
                  g=(X, P)->X, tol=1e-4, maxnit=100,
                  P = I, precon_prep! = (P, X) -> P,
                  method = "Static" )

   @unpack h = solver
   if verbose >= 2
       @printf("SADDLESEARCH:         h  =  %1.2e        <- parameters\n", h)
       @printf("SADDLESEARCH:  time | nit |  sup|∇E|_∞   \n")
       @printf("SADDLESEARCH: ------|-----|-----------------\n")
   end

   if verbose >= 4 && file!=nothing
       strlog = @sprintf("SADDLESEARCH:         h  =  %1.2e        <- parameters
SADDLESEARCH:  time | nit |  sup|∇E|_∞
SADDLESEARCH: ------|-----|-----------------\n", h)
       write(file, strlog)
       flush(file)
   end

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
   if verbose >= 2
      dt = Dates.format(now(), "HH:MM")
      @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 0, Rn)
   end
   if verbose >= 4 && file!=nothing
      dt = Dates.format(now(), "HH:MM")
      strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 0, Rn)
      write(file, strlog)
      flush(file)
   end
   if Rn <= tol
      if verbose >= 1
         println("SADDLESEARCH: $method terminates succesfully after $(nit) iterations.")
      end
      if verbose >= 4 && file!=nothing
         strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
         write(file, strlog)
         close(file)
      end
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

      # logging
      if verbose >= 2
         dt = Dates.format(now(), "HH:MM")
         @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, Rn)
      end
      if verbose >= 4 && file!=nothing
         dt = Dates.format(now(), "HH:MM")
         strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, Rn)
         write(file, strlog)
         flush(file)
      end
      if Rn <= tol
         if verbose >= 1
            println("SADDLESEARCH: $(method) terminates succesfully after $(nit) iterations.")
         end
         if verbose >= 4 && file!=nothing
            strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
            write(file, strlog)
            close(file)
         end
         return Xout, log, h
      end
   end

   # logging
   if verbose >= 1
      println("SADDLESEARCH: $(method) terminated unsuccesfully after $(maxnit) iterations.")
   end
   if verbose >= 4 && file!=nothing
      strlog = @sprintf("SADDLESEARCH: %s terminated unsuccesfully after %s iterations.\n", "$(method)", "$(maxnit)")
      write(file, strlog)
   end

   if verbose >= 4 && file!=nothing
      close(file)
   end
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


@with_kw type ODE12r
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
                  g=(X, P)->X, tol=1e-4, maxnit=100,
                  P = I, precon_prep! = (P, X) -> P,
                  method = "ODE" )

   @unpack threshold, rtol, C1, C2, h, hmin, extrapolate = solver
   if verbose >= 2
       @printf("SADDLESEARCH:      rtol  =  %1.2e        <- parameters\n", rtol)
       @printf("SADDLESEARCH: threshold  =  %1.2e        <- parameters\n", threshold)
       @printf("SADDLESEARCH:  time | nit |  sup|∇E|_∞   \n")
       @printf("SADDLESEARCH: ------|-----|-----------------\n")
   end

   if verbose >= 4 && file!=nothing
       strlog = @sprintf("SADDLESEARCH:      rtol  =  %1.2e        <- parameters
SADDLESEARCH: threshold  =  %1.2e        <- parameters
SADDLESEARCH:  time | nit |  sup|∇E|_∞
SADDLESEARCH: ------|-----|-----------------\n", rtol,threshold)
       write(file, strlog)
       flush(file)
   end

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
   if verbose >= 2
      dt = Dates.format(now(), "HH:MM")
      @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 0, Rn)
   end
   if verbose >= 4 && file!=nothing
      dt = Dates.format(now(), "HH:MM")
      strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 0, Rn)
      write(file, strlog)
      flush(file)
   end
   if Rn <= tol
      if verbose >= 1
         println("SADDLESEARCH: $method terminates succesfully after $(nit) iterations.")
      end
      if verbose >= 4 && file!=nothing
         strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
         write(file, strlog)
         close(file)
      end
      return Xout, log, h
   end

   # computation of the initial step
   r = norm(Fn ./ max.(abs.(X), threshold), Inf) + realmin(Float64)
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
      err = norm(e ./ max.(maximum([abs.(X) abs.(Xnew)],2), threshold), Inf) + realmin(Float64)

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
         @printf("SADDLESEARCH: invalid `extrapolate` parameter")
         if verbose >= 4 && file!=nothing
             strlog = @sprintf("SADDLESEARCH: invalid `extrapolate` parameter")
             write(file, strlog)
             close(file)
         end
         error("SADDLESEARCH: invalid `extrapolate` parameter")
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
         if verbose >= 2
            dt = Dates.format(now(), "HH:MM")
            @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, Rn)
         end
         if verbose >= 4 && file!=nothing
            dt = Dates.format(now(), "HH:MM")
            strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, Rn)
            write(file, strlog)
            flush(file)
         end
         if Rn <= tol
            if verbose >= 1
               println("SADDLESEARCH: $(method) terminates succesfully after $(nit) iterations.")
            end
            if verbose >= 4 && file!=nothing
               strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
               write(file, strlog)
               close(file)
            end
            return Xout, log, h
         end

         # Compute a new step size.
         h = max(0.25 * h, min(4*h, h_err, h_ls))
         # log step-size analytic results
         if verbose >= 3
            println("SADDLESEARCH:      accept: new h = $h, |F| = $(Rn)")
            println("SADDLESEARCH:                hls = $(h_ls)")
            println("SADDLESEARCH:               herr = $(h_err)")
         end
         if verbose >= 4 && file!=nothing
            strlog = @sprintf("SADDLESEARCH:      accept: new h = %s, |F| = %s
SADDLESEARCH:                hls = %s
SADDLESEARCH:               herr = %s\n", "$h", "$(Rn)", "$(h_ls)", "$(h_err)")
            write(file, strlog)
            flush(file)
         end
      else
         # compute new step size
         h = max(0.1 * h, min(0.25 * h, h_err, h_ls))
         # log step-size analytic results
         if verbose >= 3
            println("SADDLESEARCH:      reject: new h = $h")
            println("SADDLESEARCH:               |Fnew| = $(Rnew)")
            println("SADDLESEARCH:               |Fold| = $(Rn)")
            println("SADDLESEARCH:        |Fnew|/|Fold| = $(Rnew/Rn)")
         end
         if verbose >= 4 && file!=nothing
            strlog = @sprintf("SADDLESEARCH:      reject: new h = %s
SADDLESEARCH:               |Fnew| = %s
SADDLESEARCH:               |Fold| = %s
SADDLESEARCH:        |Fnew|/|Fold| = %s\n", "$h", "$(Rnew)", "$(Rn)", "$(Rnew/Rn)")
            write(file, strlog)
            flush(file)
         end
      end

      # error message if step size is too small
      if abs(h) <= hmin
         warn("SADDLESEARCH: Step size $h too small at nit = $nit.");
         if verbose >= 4 && file!=nothing
             strlog = @sprintf("SADDLESEARCH: Step size %s too small at nit = %s.\n", "$h", "$nit")
             write(file, strlog)
             close(file)
         end
         return Xout, log, h
      end
   end

   # logging
   if verbose >= 1
      println("SADDLESEARCH: $(method) terminated unsuccesfully after $(maxnit) iterations.")
   end
   if verbose >= 4 && file!=nothing
      strlog = @sprintf("SADDLESEARCH: %s terminated unsuccesfully after %s iterations.\n", "$(method)", "$(maxnit)")
      write(file, strlog)
   end

   if verbose >= 4 && file!=nothing
      close(file)
   end
   return Xout, log, h
end

@with_kw type momentum_descent
   h::Float64 = 1e-1
   b = 1e-1
   fd_scheme = :central
   redistrib = :canonical
end

function odesolve(solver::momentum_descent, f, df, X0::Vector{Float64},
                  log::IterationLog;
                  file = nothing,
                  verbose = 1,
                  g=(X, P)->X, tol=1e-4, maxnit=100,
                  P = I, precon_prep! = (P, X) -> P,
                  method = "Momentum Descent" )

   # @unpack h, b, finite_diff = solver
   @unpack h, b, fd_scheme, redistrib= solver


   if verbose >= 2
       @printf("SADDLESEARCH:         h  =  %1.2e        <- parameters\n", h)
       @printf("SADDLESEARCH:  time | nit |  sup|∇E|_∞   \n")
       @printf("SADDLESEARCH: ------|-----|-----------------\n")
   end

   if verbose >= 4 && file!=nothing
       strlog = @sprintf("SADDLESEARCH:         h  =  %1.2e        <- parameters
SADDLESEARCH:  time | nit |  sup|∇E|_∞
SADDLESEARCH: ------|-----|-----------------\n", h)
       write(file, strlog)
       flush(file)
   end

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
   if verbose >= 2
      dt = Dates.format(now(), "HH:MM")
      @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 0, Rn)
   end
   if verbose >= 4 && file!=nothing
      dt = Dates.format(now(), "HH:MM")
      strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 0, Rn)
      write(file, strlog)
      flush(file)
   end
   if Rn <= tol
      if verbose >= 1
         println("SADDLESEARCH: $method terminates succesfully after $(nit) iterations.")
      end
      if verbose >= 4 && file!=nothing
         strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
         write(file, strlog)
         close(file)
      end
      return Xout, log, h
   end

   Xnew = g(X + h * Fn, P)

   # return force
   Pnew = precon_prep!(P, Xnew)
   Fnew, Rnew, ndE, _ = f(Xnew, Pnew, 1)

   numdE += ndE

   X, Fn, Rn, P = Xnew, Fnew, Rnew, Pnew
   push!(Xout, X) # store X
   push!(log, numE, numdE, Rn) # residual, store history

   # logging
   if verbose >= 2
      dt = Dates.format(now(), "HH:MM")
      @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 1, Rn)
   end
   if verbose >= 4 && file!=nothing
      dt = Dates.format(now(), "HH:MM")
      strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 1, Rn)
      write(file, strlog)
      flush(file)
   end
   if Rn <= tol
      if verbose >= 1
         println("SADDLESEARCH: $(method) terminates succesfully after 1 Euler iteration.")
      end
      if verbose >= 4 && file!=nothing
         strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after 1 Euler iteration.\n", "$(method)")
         write(file, strlog)
         close(file)
      end
      return Xout, log, h
   end

   if b == nothing
      dFn = -df(X, P)
      Λ, _ = eig(dFn)
      λmax  = Λ[findmax(real(Λ))[2]]
      _, b = stability(λmax)
      b = b*.5

      h = 1.0; it = 1; it_max = 100
      while (it<=it_max && !minimum([criterion(fd_scheme, λ*h*h, b*h) for λ in Λ[real(Λ).>0.]]))
         h = h/2
         it+=1
      end
   end

   @printf("b = %1.2e, h = %1.2e\n",b, h)

   for nit = 2:maxnit
      # if b == nothing
      #    if mod(nit, 50) == 0
      #       dFn = -df(X, P)
      #       Λ, _ = eig(dFn)
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
      if verbose >= 2
         dt = Dates.format(now(), "HH:MM")
         @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, Rn)
      end
      if verbose >= 4 && file!=nothing
         dt = Dates.format(now(), "HH:MM")
         strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, Rn)
         write(file, strlog)
         flush(file)
      end
      if Rn <= tol
         if verbose >= 1
            println("SADDLESEARCH: $(method) terminates succesfully after $(nit) iterations.")
         end
         if verbose >= 4 && file!=nothing
            strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
            write(file, strlog)
            close(file)
         end
         return Xout, log, h
      end
   end

   # logging
   if verbose >= 1
      println("SADDLESEARCH: $(method) terminated unsuccesfully after $(maxnit) iterations.")
   end
   if verbose >= 4 && file!=nothing
      strlog = @sprintf("SADDLESEARCH: %s terminated unsuccesfully after %s iterations.\n", "$(method)", "$(maxnit)")
      write(file, strlog)
   end

   if verbose >= 4 && file!=nothing
      close(file)
   end
   return Xout, log, h
end
