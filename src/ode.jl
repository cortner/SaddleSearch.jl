@with_kw type ode23
   atol::Float64 = 1e-6
   rtol::Float64 = 1e-3
   adapt_rtol::Bool = false
end

function odesolve(solver::ode23, f, x0::Vector{Float64}, N::Int,
                  log::IterationLog, method;
                  g=x->x, tol_res=1e-4, maxnit=100 )
   @unpack atol, rtol, adapt_rtol = solver
   @unpack verbose = method

   t0 = 0
   atol0 = atol
   rtol0 = rtol

   threshold = atol/rtol

   t = t0
   x = x0[:]

   tout = []
   xout = []

   numdE, numE = 0, 0

   # computation of the initial step
   s1, _ = f(t, x)
   if adapt_rtol; rtol = min(rtol0 * norm(s1, Inf), rtol0); end
   r = norm(s1./max(abs(x),threshold),Inf) + realmin(Float64)
   h = 0.8*rtol^(1/3)/r
   numdE += N

   for nit = 0:maxnit
      hmin = 16*eps(Float64)*abs(t)

      abs(h) < hmin ? h = hmin: h = h

      s2, _ = f(t+h*0.5, x+h*0.5*s1)
      s3, _ = f(t+h*0.75, x+h*0.75*s2)
      tnew = t + h
      xnew = x + h * (2*s1 + 3*s2 + 4*s3)./9
      s4, maxres = f(tnew, xnew)
      numdE += N*3

      # error estimation
      e = h*(-5*s1 + 6*s2 + 8*s3 - 9*s4)./72
      err = norm(e./max(max(abs(x),abs(xnew)),threshold),Inf) + realmin(Float64)

      if err <= rtol
         t = tnew
         x = xnew
         x = g(x)
         push!(tout, t)
         push!(xout, x)
         s1 = s4 # Reuse final function value to start new step.

         # atol = min(atol0 * norm(s1), atol0)
         if adapt_rtol; rtol = min(rtol0 * norm(s1, Inf), rtol0); end

         push!(log, numE, numdE, maxres)

         if verbose >= 2
            @printf("%4d |   %1.2e\n", nit, maxres)
         end
         if maxres <= tol_res
            if verbose >= 1
               println("$(typeof(method)) terminates succesfully after $(nit) iterations")
            end
            return tout, xout, log
         end
      end

      # Compute a new step size.
      h = h*min(5, 0.8*(rtol/err)^(1/3) / 2)
      if abs(h) <= hmin
         warn("Step size $h too small at t = $t.");
      end
   end

   if verbose >= 1
      println("$(typeof(method)) terminated unsuccesfully after $(maxnit) iterations.")
   end
   return tout, xout, log
end


@with_kw type ode12
   atol::Float64 = 1e-6
   rtol::Float64 = 1e-3
   adapt_rtol::Bool = true
end


function odesolve(solver::ode12, f, x0::Vector{Float64}, N::Int,
                  log::IterationLog, method;
                  g=x->x, tol_res=1e-4, maxnit=100 )

   @unpack atol, rtol, adapt_rtol = solver
   @unpack verbose = method

   t0 = 0
   atol0 = atol
   rtol0 = rtol

   threshold = atol/rtol

   t = t0
   x = x0[:]

   tout = []
   xout = []

   numdE, numE = 0, 0

   # computation of the initial step
   s1, _ = f(t, x)
   if adapt_rtol; rtol = min(rtol0 * norm(s1), rtol0); end
   r = norm(s1./max(abs(x),threshold),Inf) + realmin(Float64)
   h = 0.5 * rtol^(1/2) / r
   numdE += N

   for nit = 0:maxnit
      hmin = 16*eps(Float64)*abs(t)

      abs(h) < hmin ? h = hmin: h = h

      s2, maxres = f(t+h, x+h*s1)
      tnew = t + h
      xnew = x + h * s1

      numdE += N

      # error estimation
      e = 0.5 * h * (s2 - s1)
      err = norm(e./max(max(abs(x),abs(xnew)),threshold),Inf) + realmin(Float64)

      if err <= rtol
         t = tnew
         x = xnew
         x = g(x)
         push!(tout, t)
         push!(xout, x)
         s1 = s2
         # maxres = vecnorm(s1, Inf)

         if adapt_rtol; rtol = min(rtol0 * norm(s1, Inf), rtol0); end

         push!(log, numE, numdE, maxres)

         if verbose >= 2
            @show(nit)
            @printf("%4d |   %1.2e\n", nit, maxres)
         end
         if maxres <= tol_res
            if verbose >= 1
               println("$(typeof(method)) terminates succesfully after $(nit) iterations")
            end
            return tout, xout, log
         end
      end

      # Compute a new step size.
      h = h * min(5, 0.5*sqrt(rtol/err) )
      if abs(h) <= hmin
         warn("Step size $h too small at t = $t.");
      end
   end

   if verbose >= 1
      println("$(typeof(method)) terminated unsuccesfully after $(maxnit) iterations.")
   end
   return tout, xout, log
end



@with_kw type ODE12r
   atol::Float64 = 1e-1    # ode solver parameter
   rtol::Float64 = 1e-1    # ode solver parameter
   C1::Float64 = 1e-2      # sufficientcontraction parameter
   C2::Float64 = 2.0       # residual growth control (Inf means there is no control)
   hmin::Float64 = 1e-10   # minimal allowed step size
   maxF::Float64 = 1e3     # terminate if |Fn| > maxF * |F0|
   extrapolate::Int = 3    # extrapolation style (3 seems the most robust)
end



function odesolve(solver::ODE12r, f, x0::Vector{Float64}, N::Int,
                  log::IterationLog, method;
                  g=x->x, tol_res=1e-4, maxnit=100 )

   @unpack atol, rtol, C1, C2, hmin, extrapolate = solver
   @unpack verbose = method

   t0 = 0.0

   threshold = atol/rtol

   t = t0
   x = copy(x0)

   tout = []
   xout = []

   numdE, numE = 0, 0

   # computation of the initial step
   x = g(x)
   # push(xout, x) TODO: chck if this is the correct place to move this
   Fn, Rn = f(t, x)
   r = norm(Fn ./ max(abs.(x), threshold), Inf) + realmin(Float64)
   h = 0.5 * rtol^(1/2) / r
   h = max(h, hmin)
   numdE += N
   push!(log, numE, numdE, Rn)
   @printf("%4d |   %1.2e\n", 0, Rn)

   for nit = 0:maxnit

      tnew = t + h
      xnew = g(x + h * Fn)   # the redistribution is better done here I think
                             # that way it implicitly becomes part of `f`
                             # but it seems to make the evolution slower; need more testing!
      Fnew, Rnew = f(tnew, xnew)

      numdE += N

      # error estimation
      e = 0.5 * h * (Fnew - Fn)
      err = norm(e ./ max(max(abs(x), abs(xnew)), threshold), Inf) + realmin(Float64)

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
         error("invalid `extrapolate` parameter")
      end
      if isnan(h_ls) || (h_ls < hmin)
         h_ls = Inf
      end
      # or from the error estimate
      h_err = h * 0.5 * sqrt(rtol/err)

      if accept
         t, x, Fn, Rn = tnew, xnew, Fnew, Rnew
         push!(tout, t)
         push!(xout, x)
         push!(log, numE, numdE, Rn)

         # keeping x = g(x) here technically constitutes a bug

         if verbose >= 2
            @printf("%4d |   %1.2e\n", nit, Rn)
         end
         if Rn <= tol_res
            if verbose >= 1
               println("$(typeof(method)) terminates succesfully after $(nit) iterations")
            end
            return tout, xout, log
         end

         # Compute a new step size.
         h = max(0.25 * h, min(4*h, h_err, h_ls))
         if verbose >= 3
            println("     accept: new h = $h, |F| = $(Rn)")
            println("               hls = $(h_ls)")
            println("              herr = $(h_err)")
         end
      else
         h = max(0.1 * h, min(0.25 * h, h_err, h_ls))
         if verbose >= 3
            println("     reject: new h = $h")
            println("              |Fnew| = $(Rnew)")
            println("              |Fold| = $(Rn)")
            println("       |Fnew|/|Fold| = $(Rnew/Rn)")
         end
      end

      if abs(h) <= hmin
         warn("Step size $h too small at t = $t.");
         return tout, xout, log
      end
   end

   if verbose >= 1
      println("$(typeof(solver)) terminated unsuccesfully after $(maxnit) iterations.")
   end

   # println("DEBUG")
   # Fold, _ = f(t, x)
   # println("  |Fold| = ", norm(Fold, Inf))
   # for h in (1e-1, 1e-2, 1e-3, 1e-4)
   #    Fnew, _ = f(t, x + h * Fold)
   #    println("   |Fnew(h)| = ", norm(Fnew, Inf))
   # end

   return tout, xout, log
end

# function odesolve(solver::ode12var, f, x0::Vector{T},
#                   log::IterationLog, method;
#                   g=x->x, tol_res=1e-4, maxnit=100 )
#
#    @unpack atol, rtol, adapt_rtol = solver
#    @unpack verbose = method
#
#    t0 = 0
#    atol0 = atol
#    rtol0 = rtol
#
#    threshold = atol/rtol
#
#    t = t0
#    x = x0[:]
#
#    tout = []
#    xout = []
#
#    numdE, numE = 0, 0
#
#    # computation of the initial step
#    s1, _ = f(t, x)
#    if adapt_rtol; rtol = min(rtol0 * norm(s1), rtol0); end
#    r = [norm(s1[i]./max(abs(x[i]),threshold),Inf) + realmin(Float64) for i=1:length(x)]
#    h = [0.5 * rtol^(1/2) / r[i] for i=1:length(r)]
#    rtol = [rtol for xi in x]
#    numdE += N
#
#    nit = 0
#    while nit <= maxnit
#       x = g(x)
#
#       hmin = 16*eps(Float64).*abs(t)
#
#       xtemp = copy(x)
#       ttemp = copy(t)
#
#       unchanged =  find(x->x==0, sum( cat(2,xtemp - x...), 1 ) )
#       while any(xtemp != x) || nit <= maxnit
#          [abs(h[i]) < hmin ? h[i] = hmin: h[i] = h[i] for i in unchanged]
#
#          s2, maxres = f(t+h, x+h.*s1)
#          tnew = t + h
#          xnew = x + h .* s1
#
#          numdE += N
#
#          # error estimation
#          e = 0.5 * h .* (s2 - s1)
#          err = [norm(ei./max(max(abs(x),abs(xnew)),threshold),Inf) + realmin(Float64) for ei in e]
#
#          for i in unchanged
#             if err[i] <= rtol
#                ttemp[i] = tnew[i]
#                xtemp[i] = xnew[i]
#             end
#          end
#
#          # Compute a new step sizes.
#          unchanged =  find(x->x==0, sum( cat(2,xtemp - x...), 1 ) )
#          [h[i] = h[i] * min(5, 0.5*sqrt(rtol[i]/err[i]) ) for i in unchanged]
#          for i in unchanged
#             if abs(h[i]) <= hmin[i]
#                warn("Step size $h[i] too small at t = $t[i] and node = $i.");
#             end
#          end
#
#          nit += 1
#       end
#       t = ttemp
#       x = xtemp
#       push!(tout, t)
#       push!(xout, x)
#       s1 = s2
#       # maxres = vecnorm(s1, Inf)
#
#       if adapt_rtol; [rtol[i] = min(rtol0 * norm(s1[i], Inf), rtol0) for i=1:length(rtol)]; end
#
#       push!(log, numE, numdE, maxres)
#
#       if verbose >= 2
#          @printf("%4d |   %1.2e\n", nit, maxres)
#       end
#       if maxres <= tol_res
#          if verbose >= 1
#             println("$(typeof(method)) terminates succesfully after $(nit) iterations")
#          end
#          return tout, xout, log
#       end
#
#       # Compute a new step size.
#       h = h .* min(5, 0.5*sqrt(rtol./err) )
#       for i=1:length(h)
#          if abs(h[i]) <= hmin[i]
#             warn("Step size $h[i] too small at t = $t[i] and node = $i.");
#          end
#       end
#    end
#
#    if verbose >= 1
#       println("$(typeof(method)) terminated unsuccesfully after $(maxnit) iterations.")
#    end
#    return tout, xout, log
# end
