

function bs23(f, x0::Vector{Float64}, N::Int, log::IterationLog, method;
                g=x->x, atol=1e-6, rtol=1e-3, tol_res=1e-4, maxnit=100,
                adapt_rtol = false )
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
   h = 0.8*rtol^(1/3)/r
   numdE += N

   for nit = 0:maxnit
      hmin = 16*eps(Float64)*abs(t)

      abs(h) < hmin ? h = hmin: h = h

      s2, _ = f(t+h*0.5, x+h*0.5*s1)
      s3, _ = f(t+h*0.75, x+h*0.75*s2)
      tnew = t + h
      xnew = x + h * (2*s1 + 3*s2 + 4*s3)./9
      s4, _ = f(tnew, xnew)
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
         s1, maxres = f(t,x) # Reuse final function value to start new step.
         numdE += N

         # atol = min(atol0 * norm(s1), atol0)
         if adapt_rtol; rtol = min(rtol0 * norm(s1), rtol0); end

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





function ode12(f, x0::Vector{Float64}, N::Int, log::IterationLog, method;
                g=x->x, atol=1e-6, rtol=1e-3, tol_res=1e-4, maxnit=100,
                adapt_rtol = true )
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

      s2, _ = f(t+h, x+h*s1)
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
         maxres = vecnorm(s1, Inf)

         if adapt_rtol; rtol = min(rtol0 * norm(s1), rtol0); end

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
