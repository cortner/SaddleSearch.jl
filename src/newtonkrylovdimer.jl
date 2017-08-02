using CTKSolvers: parab3p

export NK, blocklanczos


"""
`pushcol(V::Matrix, v::Vector) -> V`:
append the vector `v` as a column to `V`
"""
function pushcol(V::Matrix, v::Vector)
   rows, cols = size(V)
   V = V[:]
   append!(V, v)
   return reshape(V, rows, cols+1)
end


"""
`sorted_eig(A::Union{SymTridiagonal, Symmetric})`:
diagonalise A, and return sorted eigenvalues and analogously sorted eigenvectors
"""
function sorted_eig(A::Union{SymTridiagonal, Symmetric})
   D, Q = eig(A)
   I = sortperm(D)
   return D[I], Q[:, I]
end

nkdualnorm(P, f) = norm(f)

"""
`orthogonalise(w, V::Matrix, P)`:
Assuming V is P-orthogonal, P-orthogonalise w to V and return
"""
function orthogonalise(w, V::Matrix, P)
   # make w orthogonal to all columns of V
   for i = 1:size(V,2)
      w -= dot(w, P, V[:,i]) * V[:,i]
   end
   return w
end

"""
`basic_dirder(f, f0, xc, v, h)`:
a basic directional derivative operator to mimic the ∇²E(x) * v action
"""
function basic_dirder(f, f0, xc, v, h)
   nv = norm(v, Inf)
   h = h / nv
   return (f(xc + h * v) - f0) / h
end

ctk_dirder(f, f0, xc, v, h) =  CTKSolvers.dirder(xc, v, f, f0, h = h)

"""
`appendkrylov(V, AxV, Y, v, Hmul, P)`:
appends [V v], [AxV A*v], [Y P \ A*v]
"""
function appendkrylov(V, AxV, Y, v, Hmul, P)
   V   = pushcol(V,   v)
   AxV = pushcol(AxV, Hmul(v))
   Y   = pushcol(Y,   P \ AxV[:, end])
   return V, AxV, Y
end


# TODO: generalise dcg_index1 to allow arbitrary transformations of the
#       spectrum, and probably also arbitrary right-hand sides
# TODO: look into  re-orthogonalising

"""
blocklanczos(f0, f, xc, errtol, kmax, v1=copy(f0); P = I, reorth = true)
   -> x, λ, v

An iterative solver for
   H' u = - ∇E(x)
where -- if H = ∇^2 E(x) = Q D Q'
   H' = Q D' Q    and    D' = diag(-|λ₁|, |λ₂|, …)

This code is inspired by [insert REF] and by
   http://www4.ncsu.edu/~ctk/newton/SOLVERS/nsoli.m
(see also [CTKSolvers.jl](https://github.com/cortner/CTKSolvers.jl))

`blocklanczos` first uses a (possibly preconditioned) lanczos method to compute
a reasonable approximation to H in the form H = P V T V' P. Then it diagonalises
T = Q D Q, replaces D with D' as above. This gives an approximation H̃' to H'.
After each iteration of the lanzcos method we check whether
   pinv(H̃') * (-∇E(xc))
is a solution of sufficiently high accuracy.

In the lanzcos iterations the matrix vector product H * u
is replaced with the finite-difference operation (∇E(xc + h u) - ∇E(xc))/h;
see also `basic_dirder` and `CTKSolvers.dirder`

## Required Parameters

* f0 : ∇E(xc)
* f : function to evaluate ∇E
* xc : current point
* errtol : ????
* kmax : maximum number of lanzcos iterations

## KW parameters

* V0 : initial subspace (default: [∇E(xc)])
* P : preconditioner (default: I;  needs to define `*` and `\` )
* b : right-hand side (default: - f0)
* eigatol, eigrtol: tolerance on the first eigenvalue
* debug: show debug information (true/false)
* h : finite-difference parameter
* dirder : function to compute the directional derivative; see
         `basic_dirder` for format
<!-- * reorth : `true` to re-orthogonalise -->

## Returns

* x : approximate solution
* λ : smallest ritz value
* v : associated approximate eigenvector
"""
function blocklanczos( f0, f, xc, errtol, kmax;
                     P = I, b = - f0,
                     V0 = reshape(P \ b, length(b), 1),
                     eigatol = 1e-1, eigrtol = 1e-1,
                     debug = false, h = 1e-7,
                     dirder = basic_dirder,
                     Hmul = z -> dirder(f, f0, xc, z, h),
                     ORTHTOL = 1e-12 )

   # initialise some variables
   d = length(f0)      # problem dimension
   @assert kmax <= d
   numf = 0            # count f evaluations
   isnewton = false    # remember whether the output is a newton direction

   # allocate Krylov subspace and more
   V = zeros(d,0)      # store the Krylov basis
   AxV = zeros(d, 0)   # store A vⱼ
   Y = zeros(d, 0)     # store P \ A vⱼ

   # initialise Krylov subspace
   for j = 1:size(V0, 2)
      vj = orthogonalise(V0[:,j], V, P)
      nrmvj = norm(P, vj)
      if nrmvj > ORTHTOL
         V, AxV, Y = appendkrylov(V, AxV, Y, vj/nrmvj, Hmul, P)
         numf += 1
      else
         warn("a column of V0 is linearly dependent on the rest so I am skipping it")
      end
   end

   # prepare for the Block-Lanczos loop
   j = 1
   x = zeros(d)
   vmin = zeros(d)
   λ = minimum(eigvals(Symmetric(V'*AxV)))

   # start the block-lanczos loop; when we have kmax v-vectors we stop
   while size(V, 2) <= kmax

      # We have V, AxV and Y = P \ AxV  available, so we can immediately solve
      # the projected linear system and eigenvalue problem
      #
      # there is probably an elegant way to assemble the projected
      # linear system, but for now we just do it brute-force:
      #    A vj = P V T V' P vj
      #    yj = P \ A qj = V T  V' P vj
      #    V' P Y = V' P V T V' P V = T
      n = size(V, 2)
      T = Symmetric(V' * AxV)       # T = V' * P * Y = V' * AV
      #  TODO: could replace above with with [dot(V[:,a], P, Y[:,b] for a=1:n, b=1:n]

      # make the index-1 transformation
      D, Q = sorted_eig(T)
      E = [ - abs(D[1]); abs(D[2:end]) ]
      vmin, vmin_old = V * Q[:,1], vmin    # smallest (hopefully unstable) eigenmode
      # residual estimate for the old x    >>> TODO: should we switch to P^{-1}-norm?
      res = norm( P * (V * (Q * (E .* (Q' * (V' * (P * x)))))) - b )
      # new x and λ (remember the old)
      g = Q * (E .\ (Q' * (V' * b)))
      x, x_old = V * g, x
      λ, λ_old = D[1], λ
      # if E == D then A' = A hence we can estimate the *actual* and *current* residual
      if debug; @show j, λ; end
      if (isnewton = (E == D))
         res = norm( AxV * g - b )    # TODO: should we switch to P^{-1}-norm?
      end
      if debug; @show abs(λ - λ_old), res/norm(b); end
      # check for termination
      if res < errtol && (λ < 0 || abs(λ - λ_old) < eigatol + eigrtol * abs(λ))
         return x, λ, vmin, numf, isnewton
      end

      # add the next Krylov vector
      w = orthogonalise(Y[:, j], V, P)
      nrmw = norm(P, w)
      if nrmw > ORTHTOL
         V, AxV, Y = appendkrylov(V, AxV, Y, w/nrmw, Hmul, P)
         numf += 1
      end
      j += 1
      if j > size(V,2)   # if we have no new vector left, then we are in trouble!
         # TODO: deal with this somehow, probably by adding a random vector!
         error("I don't know yet how to deal with the case p = 0")
      end
   end
   # if we are here it means that kmax is reached, i.e. we terminate with
   # warning or error >>> TODO: return to how to best handle this?!
   # warn("`dcg_index1` did not converge within kmax = $(kmax) iterations")
   return x, λ, vmin, numf, isnewton
end



"""
Newton-Krylov based saddle search method
"""
@with_kw type NK
   tol::Float64 = 1e-5
   maxnumdE::Int = 1000
   len::Float64 = 1e-7
   precon = I
   precon_prep! = (P, x) -> P
   verbose::Int = 1
   krylovinit::Symbol = :res  # allow res, rand, rot, resrot
   maxstep::Float64 = Inf
   eigatol::Float64 = 1e-1
   eigrtol::Float64 = 1e-1
end


function run!{T}(method::NK, E, dE, x0::Vector{T},
                  v0::Vector{T} = rand(T, length(x0)) )
   # get parameters
   @unpack tol, maxnumdE, len, verbose, krylovinit, maxstep,
      eigatol, eigrtol = method
   precon = x -> method.precon_prep!(method.precon, x)
   debug = verbose > 2

   # initialise some more parameters; TODO: move these into NK?
   d = length(x0)
   eta = etamax = 0.5   # 0.9
   gamma = 0.9
   kmax = min(40, d)
   Carmijo = 1e-4
   α_old = α = 1.0

   # evaluate the initial residual
   x = copy(x0)
   v = copy(v0)
   f0 = dE(x)
   numdE = 1
   res = norm(f0, Inf)

   P = precon(x)
   fnrm = nkdualnorm(P, f0)
   fnrmo = 1.0
   itc = 0

   while res > tol && numdE < maxnumdE
      rat = fnrm / fnrmo   # TODO: is this unused?
      fnrmo = fnrm         # TODO: probably move this to where fnrm is updated!
      itc += 1

      if debug; @show dot(f0, v); end 
      # compute the (modified) Newton direction
      if krylovinit == :res
         V0 = reshape(- P \ f0, d, 1)
      elseif krylovinit == :rand
         V0 = reshape(P \ rand(d), d, 1)
      elseif krylovinit == :rot
         V0 = reshape(v, d, 1)
      end
      p, λ, v, inner_numdE, isnewton =
            blocklanczos(f0, dE, x, eta * norm(f0), kmax;
                         P = P, b = - f0, V0 = V0, debug = (verbose >= 3),
                         h = len, eigatol = eigatol, eigrtol = eigrtol)
      numdE += inner_numdE
      @show isnewton

      # ~~~~~~~~~~~~~~~~~~ LINESEARCH ~~~~~~~~~~~~~~~~~~~~~~
      if isnewton
         iarm = 0
         α = αt = 1.0
         xt = x + αt * p
         ft = dE(xt)
         numdE += 1
         nf0 = nkdualnorm(P, f0)
         nft = nkdualnorm(P, ft)

         αm = 1.0  # these have no meaning; just allocations
         nfm = 0.0

         while nft > (1 - Carmijo * αt) * nf0
            if iarm == 0
               α *= 0.5
            else
               α = parab3p( αt, αm, nf0^2, nft^2, nfm^2; sigma0 = 0.1, sigma1 = 0.5 )
            end
            αt, αm, nfm = α, αt, nft
            if αt < 1e-8   # TODO: make this a parameter
               error(" Armijo failure, step-size too small")
            end

            xt = x + αt * p
            ft = dE(xt)
            numdE += 1
            nft = nkdualnorm(P, ft)
            iarm += 1
         end
         α_old = αt
      else
         # if we are here, then p is not a newton direction (i.e. an e-val was
         # flipped) in this case, we do something very crude:
         #   take same step as before, then do one line-search step
         #   and pick the better of the two.
         αt = 0.66 * α_old  # probably can do better by re-using information from dcg_...
         αt = min(αt, maxstep / norm(p, Inf))
         xt = x + αt * p
         ft = dE(xt)
         nft = nkdualnorm(P, ft)
         numdE += 1
         # find a root: g(t) = (1-t) f0⋅p + t ft ⋅ p = 0 => t = f0⋅p / (f0-ft)⋅p
         #    if (f0-ft)⋅p is very small, then simply use xt as the next step.
         #    if it is large enough, then take just one iteration to get a root
         if abs(dot(f0 - ft, P, p)) > 1e-4   #  TODO: make this a parameter
            t = dot(f0, P, p) / dot(f0 - ft, P, p)
            t = max(t, 0.1)    # don't make too small a step
            t = min(t, 4 * t)  # don't make too large a step
            αt, αm, nfm, fm = (t*αt), αt, nft, ft
            αt = min(αt, maxstep / norm(p, Inf))
            xt = x + αt * p
            ft = dE(xt)
            numdE += 1
            nft = nkdualnorm(P, ft)
            # if the line-search step is worse than the initial trial, then
            # we revert
            if abs(dot(ft, P, p)) > abs(dot(fm, P, p))
               αt, xt, nft, ft = αm, x + αm * p, nfm, fm
            end
         end
      end
      if verbose > 3; @show αt; end
      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      # update current configuration and preconditioner
      x, f0, fnrm = xt, ft, nft
      P = precon(x)
      res = norm(f0, Inf)
      fnrm = nkdualnorm(P, f0)     # should be the dual norm!
      rat = fnrm/fnrmo

      if verbose > 3; @show λ, res; end

      if res <= tol
         return x, numdE
      end

      # ---------------------------------------------------------
      # Adjust eta as per Eisenstat-Walker.   # TODO: make this a flag!
      # TODO: check also the we are in the index-1 regime (what do we do if
      # not? probably reset eta?)
      etaold = eta
      etanew = gamma * rat^2
      if gamma * etaold^2 > 0.1
         etanew = max(etanew, gamma * etaold^2)
      end
      eta = min(etanew, etamax)
      eta = max(eta, 0.5 * tol / fnrm)
      # ---------------------------------------------------------

   end

   if verbose > 1
      warn("NK did not converge within the maximum number of dE evaluations")
   end
   return x, numdE
end
