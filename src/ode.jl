

@with_kw type ODE12r
   rtol::Float64 = 1e-1    # ode solver parameter
   threshold::Float64 = 1.0   # threshold for error estimate; we want to get rid of this
   C1::Float64 = 1e-2      # sufficientcontraction parameter
   C2::Float64 = 2.0       # residual growth control (Inf means there is no control)
   hmin::Float64 = 1e-10   # minimal allowed step size
   maxF::Float64 = 1e3     # terminate if |Fn| > maxF * |F0|
   extrapolate::Int = 3    # extrapolation style (3 seems the most robust)
end



function odesolve(solver::ODE12r, f, x0::Vector{Float64}, log::IterationLog;
                  verbose = 1,
                  g=(x, P)->x, tol_res=1e-4, maxnit=100,
                  P = I, precon_prep! = (P, x) -> P,
                  method = "ODE" )

   @unpack threshold, rtol, C1, C2, hmin, extrapolate = solver

   if verbose>=4
       dt = Dates.format(now(), "d-m-yyyy_HH:MM")
       file = open("log_$(dt).txt", "w")
   end

   x = copy(x0)
   P = precon_prep!(P, x)

   xout = []

   numdE, numE = 0, 0

   # computation of the initial step
   x = g(x, P)
   P = precon_prep!(P, x)
   Fn, Rn, ndE = f(x, P, 0)
   numdE += ndE

   push!(xout, x)
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
   if Rn <= tol_res
      if verbose >= 1
         println("SADDLESEARCH: $method terminates succesfully after $(nit) iterations")
      end
      if verbose >= 4
         strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
         write(file, strlog)
         close(file)
      end
      return xout, log
   end

   r = norm(Fn ./ max.(abs.(x), threshold), Inf) + realmin(Float64)
   h = 0.5 * rtol^(1/2) / r
   h = max(h, hmin)

   for nit = 1:maxnit

      xnew = g(x + h * Fn, P)   # the redistribution is better done here I think
                             # that way it implicitly becomes part of `f`
                             # but it seems to make the evolution slower; need more testing!
      Pnew = precon_prep!(P, xnew)
      Fnew, Rnew, ndE = f(xnew, Pnew, nit)

      numdE += ndE

      # error estimation
      e = 0.5 * h * (Fnew - Fn)
      err = norm(e ./ max.(maximum([abs.(x) abs.(xnew)],2), threshold), Inf) + realmin(Float64)

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
         h_ls = h * norm(Fn)^2 / dot(Fn, y)
      elseif extrapolate == 2   # F(xn + h Fn) ⋅ F{n+1} ~ 0
         # h_ls = h * dot(Fn, y) / (norm(y)^2 + 1e-10)
         h_ls = h * dot(Fn, Fnew) / (dot(Fn, y) + 1e-10)
      elseif extrapolate == 3   # min | F(xn + h Fn) |
         h_ls = h * dot(Fn, y) / (norm(y)^2 + 1e-10)
      else
         @printf("SADDLESEARCH: invalid `extrapolate` parameter")
         if verbose >=4
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
         x, Fn, Rn, P = xnew, Fnew, Rnew, Pnew

         push!(xout, x)
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
         if Rn <= tol_res
            if verbose >= 1
               println("SADDLESEARCH: $(method) terminates succesfully after $(nit) iterations")
            end
            if verbose >= 4
               strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations\n", "$(method)", "$(nit)")
               write(file, strlog)
               close(file)
            end
            return xout, log
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
         if verbose >=4
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
         return xout, log
      end
   end

   if verbose >= 1
      println("SADDLESEARCH: $(typeof(solver)) terminated unsuccesfully after $(maxnit) iterations.")
   end
   if verbose >=4
      strlog = @sprintf("SADDLESEARCH: %s terminated unsuccesfully after %s iterations.\n", "$(typeof(solver))", "$(maxnit)")
      write(file, strlog)
   end

   if verbose >=4
      close(file)
   end
   return xout, log
end
