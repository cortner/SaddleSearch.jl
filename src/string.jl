
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
# @with_kw type StaticString
#    precon_scheme = localPrecon()
#    alpha::Float64
#    path_traverse = serial()
#    # ------ shared parameters ------
#    tol_res::Float64 = 1e-5
#    maxnit::Int = 1000
#    verbose::Int = 2
# end

function run!{T}(method::StaticString, E, dE, x0::Vector{T})
   # read all the parameters
   @unpack alpha, tol, maxnit, precon_scheme, path_traverse,
            verbose = method
   @unpack direction = path_traverse
   # initialise variables
   x = copy(x0)
   nit = 0
   numdE, numE = 0, 0
   log = PathLog()

   xref = redistribute(ref(x), x, precon_scheme)
   Fn, Rn, ndE = forces(precon_scheme, x, xref, dE, direction(length(x), nit))
   numdE += ndE

   # and just start looping
   if verbose >= 2
      @printf("SADDLESEARCH:  time | nit |  sup|∇E|_∞   \n")
      @printf("SADDLESEARCH: ------|-----|-----------------\n")
   end
   for nit = 1:maxnit
      # redistribute
      xnew = redistribute(xref + alpha * Fn, x, precon_scheme)

      # return force
      Fn, Rn, ndE = forces(precon_scheme, x, xnew, dE, direction(length(x), nit))
      numdE += ndE

      # residual, store history
      push!(log, numE, numdE, Rn)

      if verbose >= 2
         dt = Dates.format(now(), "HH:MM")
         @printf("SADDLESEARCH: %s |%4d |   %1.2e\n", dt, nit, res)
      end
      if Rn <= tol
         if verbose >= 1
            println("SADDLESEARCH: StaticString terminates succesfully after $(nit) iterations")
         end
         return set_ref!(x, xnew), log
      end
      xref = xnew
   end
   if verbose >= 1
      println("SADDLESEARCH: StaticString terminated unsuccesfully after $(maxnit) iterations.")
   end
   return set_ref!(x, xref), log
end
