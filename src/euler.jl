
"""
`StaticString`: a preconditioning variant of the string method.

### Parameters:
* 'precon_scheme' : preconditioning method
* `alpha` : step length
* `refine_points` : number of points allowed in refinement region, negative for no refinement of path
* `ls_cond` : true/false whether to perform linesearch during the minimisation step
* `tol_res` : residual tolerance
* `maxnit` : maximum number of iterations
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
"""
@with_kw type Euler
   h::Float64 = 1e-1
end

function odesolve(solver::Euler, f, x0::Vector{Float64}, log::IterationLog;
                  verbose = 1,
                  g=(x, P)->x, tol=1e-4, maxnit=100,
                  P = I, precon_prep! = (P, x) -> P,
                  method = "Static" )

   @unpack h = solver

   if verbose >= 4
       dt = Dates.format(now(), "d-m-yyyy_HH:MM")
       file = open("log_$(dt).txt", "w")
   end

   x = copy(x0)
   P = precon_prep!(P, x)

   xout = []

   numdE, numE = 0, 0

   # initialise variables
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
   if Rn <= tol
      if verbose >= 1
         println("SADDLESEARCH: $method terminates succesfully after $(nit) iterations.")
      end
      if verbose >= 4
         strlog = @sprintf("SADDLESEARCH: %s terminates succesfully after %s iterations.\n", "$(method)", "$(nit)")
         write(file, strlog)
         close(file)
      end
      return xout, log
   end

   for nit = 1:maxnit
      # redistribute
      xnew = g(x + h * Fn, P)

      # return force
      Pnew = precon_prep!(P, xnew)
      Fnew, Rnew, ndE = f(xnew, Pnew, nit)

      numdE += ndE

      x, Fn, Rn, P = xnew, Fnew, Rnew, Pnew
      push!(xout, x)
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
         return xout, log
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
   return xout, log
end



# function run!{T}(method::StaticString, E, dE, x0::Vector{T})
#    # read all the parameters
#    @unpack alpha, tol, maxnit, precon_scheme, path_traverse,
#             verbose = method
#    @unpack direction = path_traverse
#    # initialise variables
#    x = copy(x0)
#    nit = 0
#    numdE, numE = 0, 0
#    log = PathLog()
#
#    xref = redistribute(ref(x), x, precon_scheme)
#    Fn, Rn, ndE = forces(precon_scheme, x, xref, dE, direction(length(x), nit))
#    numdE += ndE
#
#    # and just start looping
#    if verbose >= 2
#       @printf("SADDLESEARCH:  time | nit |  sup|∇E|_∞   \n")
#       @printf("SADDLESEARCH: ------|-----|-----------------\n")
#    end
#    for nit = 1:maxnit
#       # redistribute
#       xnew = redistribute(xref + alpha * Fn, x, precon_scheme)
#
#       # return force
#       Fn, Rn, ndE = forces(precon_scheme, x, xnew, dE, direction(length(x), nit))
#       numdE += ndE
#
#       # residual, store history
#       push!(log, numE, numdE, Rn)
#
#       if verbose >= 2
#          dt = Dates.format(now(), "HH:MM")
#          @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, res)
#       end
#       if Rn <= tol
#          if verbose >= 1
#             println("SADDLESEARCH: StaticString terminates succesfully after $(nit) iterations")
#          end
#          return set_ref!(x, xnew), log
#       end
#       xref = xnew
#    end
#    if verbose >= 1
#       println("SADDLESEARCH: StaticString terminated unsuccesfully after $(maxnit) iterations.")
#    end
#    return set_ref!(x, xref), log
# end
