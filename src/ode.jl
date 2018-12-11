
"""
`Euler`: simple Euler method ODE solver with fixed step length.

### Parameters:
* `h` : step length
"""
@with_kw type Euler
   h::Float64 = 1e-1
end

function odesolve(solver::Euler, f, X0::Vector{Float64}, log::IterationLog;
                  verbose = 1,
                  g=(X, P)->X, tol=1e-4, maxnit=100,
                  P = I, precon_prep! = (P, X) -> P,
                  method = "Static" )

   @unpack h = solver

   if verbose >= 4
       dt = Dates.format(now(), "d-m-yyyy_HH:MM")
       file = open("log_$(dt).txt", "w")
   end

   X = copy(X0)
   P = precon_prep!(P, X)

   Xout = [];

   numdE, numE = 0, 0

   # initialise variables
   X = g(X, P)
   P = precon_prep!(P, X)
   Fn, Rn, ndE = f(X, P, 0)
   numdE += ndE

   push!(Xout, X)
   push!(log, numE, numdE, Rn)

   if verbose >= 2
      dt = Dates.format(now(), "HH:MM")
      @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 0, Rn)
   end
   if verbose >= 4
      dt = Dates.format(now(), "HH:MM")
      strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 0, Rn)
      write(file, strlog)
      flush(file)
   end
   if Rn <= tol
      if verbose >= 1
         println("SADDLESEARCH: $method terminates succesfully after $(nit) iterations.")
      end
      if verbose >= 4
         strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
         write(file, strlog)
         close(file)
      end
      return Xout, log
   end

   for nit = 1:maxnit
      # redistribute
      Xnew = g(X + h * Fn, P)

      # return force
      Pnew = precon_prep!(P, Xnew)
      Fnew, Rnew, ndE = f(Xnew, Pnew, nit)

      numdE += ndE

      X, Fn, Rn, P = Xnew, Fnew, Rnew, Pnew
      push!(Xout, X)
      push!(log, numE, numdE, Rn) # residual, store history

      if verbose >= 2
         dt = Dates.format(now(), "HH:MM")
         @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, Rn)
      end
      if verbose >= 4
         dt = Dates.format(now(), "HH:MM")
         strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, Rn)
         write(file, strlog)
         flush(file)
      end
      if Rn <= tol
         if verbose >= 1
            println("SADDLESEARCH: $(method) terminates succesfully after $(nit) iterations.")
         end
         if verbose >= 4
            strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
            write(file, strlog)
            close(file)
         end
         return Xout, log
      end
   end

   if verbose >= 1
      println("SADDLESEARCH: $(method) terminated unsuccesfully after $(maxnit) iterations.")
   end
   if verbose >= 4
      strlog = @sprintf("SADDLESEARCH: %s terminated unsuccesfully after %s iterations.\n", "$(method)", "$(maxnit)")
      write(file, strlog)
   end

   if verbose >= 4
      close(file)
   end
   return Xout, log
end


"""
`ODE12r`: adaptive ODE solver, uses 1st and 2nd order approximations to estimate local error and find a new step length

### Parameters:
* `rtol` : relative tolerance
* `threshold` : threshold for error estimate
* `C1` : sufficient contraction parameter
* `C2` : residual growth control (Inf means there is no control)
* `hmin` : minimal allowed step size
* `maxF` : terminate if |Fn| > maxF * |F0|
* `extrapolate` : extrapolation style (3 seems the most robust)
"""


@with_kw type ODE12r
   rtol::Float64 = 1e-1
   threshold::Float64 = 1.0
   C1::Float64 = 1e-2
   C2::Float64 = 2.0
   hmin::Float64 = 1e-10
   maxF::Float64 = 1e3
   extrapolate::Int = 3
end

function odesolve(solver::ODE12r, f, X0::Vector{Float64}, log::IterationLog;
                  verbose = 1,
                  g=(X, P)->X, tol=1e-4, maxnit=100,
                  P = I, precon_prep! = (P, X) -> P,
                  method = "ODE" )

   @unpack threshold, rtol, C1, C2, hmin, extrapolate = solver

   if verbose >= 4
       dt = Dates.format(now(), "d-m-yyyy_HH:MM")
       file = open("log_$(dt).txt", "w")
   end

   X = copy(X0)
   P = precon_prep!(P, X)

   Xout = [];

   numdE, numE = 0, 0

   # computation of the initial step
   X = g(X, P)
   P = precon_prep!(P, X)
   Fn, Rn, ndE, _ = f(X, P, 0)
   numdE += ndE

   push!(Xout, X)
   push!(log, numE, numdE, Rn)

   if verbose >= 2
      dt = Dates.format(now(), "HH:MM")
      @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 0, Rn)
   end
   if verbose >= 4
      dt = Dates.format(now(), "HH:MM")
      strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, 0, Rn)
      write(file, strlog)
      flush(file)
   end
   if Rn <= tol
      if verbose >= 1
         println("SADDLESEARCH: $method terminates succesfully after $(nit) iterations.")
      end
      if verbose >= 4
         strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
         write(file, strlog)
         close(file)
      end
      return Xout, log
   end

   r = norm(Fn ./ max.(abs.(X), threshold), Inf) + realmin(Float64)
   h = 0.5 * rtol^(1/2) / r
   h = max(h, hmin)

   for nit = 1:maxnit

      Xnew = g(X + h * Fn, P)   # that way it implicitly becomes part
                                # of `f`
                                # but it seems to make the evolution slower; need more testing!
      Pnew = precon_prep!(P, Xnew)
      Fnew, Rnew, ndE, dot_P = f(Xnew, Pnew, nit)

      numdE += ndE

      # error estimation
      e = 0.5 * h * (Fnew - Fn)
      err = norm(e ./ max.(maximum([abs.(X) abs.(Xnew)],2), threshold), Inf) + realmin(Float64)

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
         if verbose >= 4
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

         push!(Xout, X)
         push!(log, numE, numdE, Rn)

         if verbose >= 2
            dt = Dates.format(now(), "HH:MM")
            @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, Rn)
         end
         if verbose >= 4
            dt = Dates.format(now(), "HH:MM")
            strlog = @sprintf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, Rn)
            write(file, strlog)
            flush(file)
         end
         if Rn <= tol
            if verbose >= 1
               println("SADDLESEARCH: $(method) terminates succesfully after $(nit) iterations.")
            end
            if verbose >= 4
               strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
               write(file, strlog)
               close(file)
            end
            return Xout, log
         end

         # Compute a new step size.
         h = max(0.25 * h, min(4*h, h_err, h_ls))
         if verbose >= 3
            println("SADDLESEARCH:      accept: new h = $h, |F| = $(Rn)")
            println("SADDLESEARCH:                hls = $(h_ls)")
            println("SADDLESEARCH:               herr = $(h_err)")
         end
         if verbose >= 4
            strlog = @sprintf("SADDLESEARCH:      accept: new h = %s, |F| = %s\n
                               SADDLESEARCH:                hls = %s\n
                               SADDLESEARCH:               herr = %s\n",
                               "$h", "$(Rn)", "$(h_ls)", "$(h_err)")
            write(file, strlog)
            flush(file)
         end
      else
         h = max(0.1 * h, min(0.25 * h, h_err, h_ls))
         if verbose >= 3
            println("SADDLESEARCH:      reject: new h = $h")
            println("SADDLESEARCH:               |Fnew| = $(Rnew)")
            println("SADDLESEARCH:               |Fold| = $(Rn)")
            println("SADDLESEARCH:        |Fnew|/|Fold| = $(Rnew/Rn)")
         end
         if verbose >= 4
            strlog = @sprintf("SADDLESEARCH:      reject: new h = %s\n
                               SADDLESEARCH:               |Fnew| = %s\n
                               SADDLESEARCH:               |Fold| = %s\n
                               SADDLESEARCH:        |Fnew|/|Fold| = %s\n",
                               "$h", "$(Rnew)", "$(Rn)", "$(Rnew/Rn)")
            write(file, strlog)
            flush(file)
         end
      end

      if abs(h) <= hmin
         warn("SADDLESEARCH: Step size $h too small at nit = $nit.");
         if verbose >= 4
             strlog = @sprintf("SADDLESEARCH: Step size %s too small at nit = %s.\n", "$h", "$nit")
             write(file, strlog)
             close(file)
         end
         return Xout, log
      end
   end

   if verbose >= 1
      println("SADDLESEARCH: $(method) terminated unsuccesfully after $(maxnit) iterations.")
   end
   if verbose >= 4
      strlog = @sprintf("SADDLESEARCH: %s terminated unsuccesfully after %s iterations.\n", "$(method)", "$(maxnit)")
      write(file, strlog)
   end

   if verbose >= 4
      close(file)
   end
   return Xout, log
end
