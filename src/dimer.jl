

export StaticDimerMethod, run!


Base.dot{T}(x, A::UniformScaling{T}, y) = A.λ * dot(x,y)
Base.dot(x, A::AbstractMatrix, y) = dot(x, A*y)

"""
`StaticDimerMethod`: the most basic dimer variant, simply taking alternating
steps with a fixed step-size.

### Parameters:
* `a_trans` : translation step
* `a_rot` : rotation step
* `tol_trans` : translation residual
* `tol_rot` : rotation residual
* `maxnit` : maximum number of iterations
* `len` : dimer-length (i.e. distance of the two walkers)
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_rot` : true/false whether to precondition the rotation step
"""
@with_kw type StaticDimerMethod
   a_trans::Float64
   a_rot::Float64
   # ------ shared parameters ------
   tol_trans::Float64 = 1e-5
   tol_rot::Float64 = 1e-2
   maxnit::Int = 1000
   len::Float64 = 1e-2
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_rot::Bool = false
end


function run!{T}(method::StaticDimerMethod, E, dE, x0::Vector{T}, v0::Vector{T})

   # read all the parameters
   @unpack a_trans, a_rot, tol_trans, tol_rot, maxnit, len,
            precon_prep!, verbose, precon_rot = method
   P=method.precon
   # initialise variables
   x, v = copy(x0), copy(v0)
   nit = 0
   numdE, numE = 0, 0
   log = IterationLog()
   # and just start looping
   if verbose >= 2
      @printf(" nit |  |∇E|_∞    |∇R|_∞     λ   \n")
      @printf("-----|-----------------------------\n")
   end
   for nit = 0:maxnit
      # normalise v
      P = precon_prep!(P, x)
      v /= sqrt(dot(v, P, v))
      # evaluate gradients, and more stuff
      dE0 = dE(x)
      dEv = dE(x + len * v)
      numdE += 2
      Hv = (dEv - dE0) / len
      # translation and rotation residual, store history
      res_trans = vecnorm(dE0, Inf)
      p_rot = - Hv + dot(v, Hv) * (P * v)
      res_rot = vecnorm(p_rot, Inf)
      push!(log, numE, numdE, res_trans, res_rot)
      if verbose >= 2
         @printf("%4d | %1.2e  %1.2e  %4.2f  \n", nit, res_trans, res_rot, dot(v, Hv))
      end
      if res_trans <= tol_trans && res_rot <= tol_rot
         if verbose >= 1
            println("StaticDimerMethod terminates succesfully after $(nit) iterations")
            return x, v, log
         end
      end
      # translation step
      p_trans = - P \ dE0 + 2.0 * dot(v, dE0) * v
      x += a_trans * p_trans
      # rotation step
      if precon_rot
         p_rot = P \ p_rot
      end
      v += a_rot * p_rot
   end
   if verbose >= 1
      println("StaticDimerMethod terminated unsuccesfully after $(maxnit) iterations.")
      return x, v, log
   end
end
