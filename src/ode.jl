function bs23(f, tspan::Vector{Float64}, y0::Vector{Float64}; g=y->y, rtol=1e-3, atol=1e-6 )
   t0 = tspan[1]
   tfinal = tspan[end]

   tdir = sign(tfinal-t0)
   threshold = atol/rtol
   hmax = abs(0.1*(tfinal-t0));
   t = t0
   y = y0[:]

   tout = []
   yout = []

   # computation of the initial step
   s1 = f(t, y)
   r = norm(s1./max(abs(y),threshold),Inf) + realmin(Float64)
   h = tdir*0.8*rtol^(1/3)/r

   while t != tfinal
      hmin = 16*eps(Float64)*abs(t)
      abs(h) > hmax ? h = tdir*hmax : h = h
      abs(h) < hmin ? h = tdir*hmin: h = h
      # Stretch the step if t is close to tfinal.
      if 1.1*abs(h) >= abs(tfinal - t)
         h = tfinal - t
      end

      s2 = f(t+h*0.5, y+h*0.5*s1)
      s3 = f(t+h*0.75, y+h*0.75*s2)
      tnew = t + h
      ynew = y + h * (2*s1 + 3*s2 + 4*s3)./9
      s4 = f(tnew, ynew)

      # error estimation
      e = h*(-5*s1 + 6*s2 + 8*s3 - 9*s4)./72
      err = norm(e./max(max(abs(y),abs(ynew)),threshold),Inf) + realmin(Float64)

      if err <= rtol
         t = tnew
         y = ynew
         y = g(y)
         push!(tout, t)
         push!(yout, y)
         s1 = f(t,y) #s4 # Reuse final function value to start new step.
      end

      # Compute a new step size.
      h = h*min(5, 0.8*(rtol/err)^(1/3))
      if abs(h) <= hmin
         # warning(sprintf(’Step size %e too small at t = %e.\n’,h,t));
         t = tfinal;
      end
   end
   return tout, yout
end
