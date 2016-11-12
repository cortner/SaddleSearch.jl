

export SeqMinDimer



"""
`SeqMinDimer`: a sequential minimisation type dimer method

### Specific Parameters:

### Generic Parameters
* `tol_trans` : translation residual
* `tol_rot` : rotation residual
* `maxnit` : maximum number of iterations
* `len` : dimer-length (i.e. distance of the two walkers)
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `verbose` : how much information to print (0: none, 1:end of iteration, 2:each iteration)
* `precon_rot` : true/false whether to precondition the rotation step
"""
@with_kw type SeqMinDimer

   # ------ shared parameters ------
   tol_trans::Float64 = 1e-5
   tol_rot::Float64 = 1e-2
   maxn_dE::Int = 2000
   len::Float64 = 1e-3
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 2
   precon_rot::Bool = false
end



function run!{T}(method::SeqMinDimer, E, dE, x0::Vector{T}, v0::Vector{T})
   # read all the parameters
   @unpack tol_trans, tol_rot, maxnit, len,
            precon_prep!, verbose, precon_rot = method
   P = method.precon
   # initialise variables
   x, v = copy(x0), copy(v0)
   nit = 0
   numdE, numE = 0, 0
   log = IterationLog()

   # initial evaluation
   P = precon_prep!(P, x)
   v /= sqrt(dot(v, P, v))
   dE0, dEv = dE(x), dE(x + len * v)
   numdE += 2
   Hv = (dEv - dE0) / len
   dR = Hv - dot(v, Hv) * (P*v)
   res_trans = vecnorm(dE0, Inf)
   res_rot = vecnorm(dR, Inf)

   while true

      # decide whether to minimise or to translate


      #

   end
end
