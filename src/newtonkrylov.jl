using CTKSolvers: dirder

export dcg_index1, NK

function pushcol(V::Matrix, v::Vector)
   rows, cols = size(V)
   V = V[:]
   append!(V, v)
   return reshape(V, rows, cols+1)
end

function sorted_eig(A::SymTridiagonal)
   D, Q = eig(A)
   I = sortperm(D)
   return D[I], Q[:, I]
end


# TODO: generalise dcg_index1 to allow arbitrary transformations of the
#       spectrum, and probably also arbitrary right-hand sides
# TODO: look into  re-orthogonalising

"""
dcg_index1(f0, f, xc, errtol, kmax, v1=copy(f0); P = I, reorth = true)
   -> x, λ, v

An iterative solver for
   H' u = - ∇E(x)
where -- if H = ∇^2 E(x) = Q D Q'
   H' = Q D' Q    and    D' = diag(-|λ₁|, |λ₂|, …)

This code is inspired by [insert REF] and by
   http://www4.ncsu.edu/~ctk/newton/SOLVERS/nsoli.m
(see also [CTKSolvers.jl](https://github.com/cortner/CTKSolvers.jl))

`dcg_index1` first uses a (possibly preconditioned) lanzcos method to compute
a reasonable approximation to H in the form H = P V T V' P. Then it diagonalises
T = Q D Q, replaces D with D' as above. This gives an approximation H̃' to H'.
After each iteration of the lanzcos method we check whether
   pinv(H̃') * (-∇E(xc))
is a solution of sufficiently high accuracy.

In the lanzcos iterations the matrix vector product H * u
is replaced with the finite-difference operation (∇E(xc + h u) - ∇E(xc))/h;
see also `CTKSolvers.dirder`

## Required Parameters

* f0 : ∇E(xc)
* f : function to evaluate ∇E
* xc : current point
* errtol : ????
* kmax : maximum number of lanzcos iterations
* v1 : initial iterate (default: ∇E(xc))

## KW parameters

* P : preconditioner ( needs to define `*` and `\` )
* reorth : `true` to re-orthogonalise

## Returns

* x : approximate solution
* λ : smallest ritz value
* v : associated approximate eigenvector
"""
function dcg_index1(f0, f, xc, errtol, kmax;
                    P = I, b = - f0, v1 = P \ b,
                    eigatol = 1e-1, eigrtol = 1e-1,
                    debug = false )
   # allocate arrays
   d = length(f0)
   V = zeros(d, 0)     # store the Krylov basis
   AxV = zeros(d, 0)   # store A vⱼ   (as opposed to w̃ = P \ A vⱼ)
   α = Float64[]
   β = Float64[]
   numf = 0
   # initialise
     V = pushcol(  V, v1 / norm(P, v1))            # store v₁
   # dAv = dirder(xc, V[:,1], f, f0)
   dAv = (f(xc + 1e-7 * V[:,1]) - f0) * 1e7
   AxV = pushcol(AxV, dAv)  # A * v₁
   numf += 1
   w̃ = P \ AxV[:,1]                   # w̃ = P \ (A * v₁)
   push!(α, dot(w̃, P, V[:,1]))        # α₁ = <w̃, v₁>_P = <Av₁, v₁>  (TODO)
   push!(β, 0.0)                      # so the indices match Wikipedia (discard later)
   w = w̃ - α[1] * V[:,1]
   # start lanczos loop
   λ = α[1]                           # initial guess for lowest e-val
   x = zeros(d)                       # trivial initial guess for the solution
   v = zeros(d)                       # trivial guess for minimal e-vec
   for j = 2:kmax
      push!(β, norm(P, w))            # βⱼ = |w|_P
      if β[j] > 1e-7
         V = pushcol(V, w / β[j])        # vⱼ = w / βⱼ = w / |w|_P
         # make V[:,j] orthogonal to all the previous Vs
         for i = 1:j-1
            V[:,j] -= dot(V[:,j], P, V[:,i]) * V[:,i]
            if norm(P, V[:,j]) < 1e-7
              error("new lanzcos vector in range of previous space")
            end
            V[:,j] /= norm(P, V[:,j])
         end
      else
         # probably terminate
         error("need to treat this special case!")
      end

      # dAv = dirder(xc, V[:,j], f, f0)
      dAv = (f(xc + 1e-7 * V[:,j]) - f0) * 1e7
      AxV = pushcol(AxV, dAv)    # A * vⱼ
      numf += 1
      w̃ = P \ AxV[:,j]                            # w̃ = P \ (A * vⱼ)
      push!(α, dot(w̃, P, V[:,j]))                 # αⱼ = <w̃, vⱼ>_P = < Avⱼ, vⱼ>  (TODO)
      w = w̃ - α[j] * V[:,j] - β[j] * V[:, j-1]

      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # at this point, we can form V T' V' and solve for x
      T = SymTridiagonal(α, β[2:end])
      D, Q = sorted_eig(T)
      E = zeros(length(D))
      E[1] = - abs(D[1])
      E[2:end] = abs(D[2:end])
      v = V * Q[:,1]
      # we now have  A'    ~ P * V * Q *  D  * Q' * V' * P
      #         and (A')⁻¹ ~     V * Q * D⁻¹ * Q' * V'
      # (because V * P * V' = I_jxj)
      # residual estimate for the old x
      res = norm( P * (V * (Q * (E .* (Q' * (V' * (P * x)))))) - b )
      # new x and λ (remember the old)
      g = Q * (E .\ (Q' * (V' * b)))
      x, x_old = V * g, x
      λ, λ_old = D[1], λ
      # if E == D then A' = A hence we can do better to estimate the residual
      if debug; @show j, λ; end
      if E == D
         res = norm( AxV * g - b )
      end
      if debug; @show abs(λ - λ_old), res/norm(b); end
      # check for termination
      if res < errtol && (λ < 0 || bs(λ - λ_old) < eigatol + eigrtol * abs(λ))
         return x, λ, v, numf
      end
      # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   end
   # if we are here it means that kmax is reached, i.e. we terminate with
   # warning or error
   # warn("`dcg_index1` did not converge within kmax = $(kmax) iterations")
   return x, λ, v, numf
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
   verbose::Int = 2
   krylovinit::Symbol = :res  # allow res, rand, rot
end


function run!{T}(method::NK, E, dE, x0::Vector{T},
                  v0::Vector{T} = rand(T, length(x0)))
   # get parameters
   @unpack tol, maxnumdE, len, verbose, krylovinit = method
   precon = x -> method.precon_prep!(method.precon, x)


   # initialise some more parameters; TODO: move these into NK?
   d = length(x0)
   eta = etamax = 0.9
   gamma = 0.9
   kmax = min(40, d)


   # evaluate the initial residual
   x = copy(x0)
   v = copy(v0)
   f0 = dE(x)
   numdE = 1
   res = norm(f0, Inf)

   P = precon(x)
   fnrm = norm(P, f0)
   fnrmo = 1.0
   itc = 0

   while res > tol && numdE < maxnumdE
      rat = fnrm / fnrmo   # TODO: is this unused?
      fnrmo = fnrm         # TODO: probably move this to where fnrm is updated!
      itc += 1

      # compute the (modified) Newton direction
      if krylovinit == :res
         v1 = - P \ f0
      elseif krylovinit == :rand
         v1 = P \ rand(d)
      elseif krylovinit == :rot
         v1 = v
      end
      p, λ, v, inner_numdE = dcg_index1(f0, dE, x, eta * norm(f0), kmax;
                                        P = P, b = - f0, v1 = v1, debug = true)
      numdE += inner_numdE

      # for now try without linesearch
      α = min(1.0, 2.0 / res)

      # update
      x += α * p
      f0 = dE(x)
      res = norm(f0, Inf)
      fnrm = norm(P, f0)
      rat = fnrm/fnrmo

      @show λ, res

      if res <= tol
         return x, numdE
      end

      # Adjust eta as per Eisenstat-Walker.   # TODO: make this a flag!
      # TODO: check also the we are in the index-1 regime
      etaold = eta
      etanew = gamma * rat^2
      if gamma * etaold^2 > 0.1
         etanew = max(etanew, gamma * etaold^2)
      end
      eta = min(etanew, etamax)
      eta = max(eta, 0.5 * tol / fnrm)
      # ---------------------------------------------------------

   end

   warn("NK did not converge within the maximum number of dE evaluations")
   return x, numdE
end




# this code is here just for testing; should eventually be removed
function _lanczos_(A, N, v1=rand(size(A,1)); P = I, b = nothing, debug = false)
    @assert size(A,1) == size(A,2)
    @assert vecnorm(A - A', Inf) < 1e-12
    @assert N <= size(A,1)
    v1 /= norm(P, v1)
    d = size(A,1)
    V = zeros(d, N)
    T = zeros(N, N)
    α = zeros(N)
    β = zeros(N)
    V[:, 1] = v1
    w̃ = P \ (A * V[:,1])
    α[1] = dot(w̃, P, V[:,1])
    w = w̃ - α[1] * V[:,1]
    for j = 2:N
        β[j] = norm(P, w)
        if β[j] > 1e-7
            V[:,j] = w / β[j]
            # make V[:,j] orthogonal to all the previous Vs
            #  (this is probably very naive, but I don't care for now)
            for i = 1:j-1
                V[:,j] -= dot(V[:,j], P, V[:,i]) * V[:,i]
                if norm(P, V[:,j]) < 1e-7
                    error("new lanzcos vector in range of previous space")
                end
                V[:,j] /= norm(P, V[:,j])
            end
        else
            error("need to treat this special case!")
        end
        w̃ = P \ (A * V[:,j])
        α[j] = dot(w̃, P, V[:,j])

        # ===============  DEBUGGING / TESTING  ===============
        if debug
            T = spdiagm((β[2:j], α[1:j], β[2:j]), (-1,0,1), j, j)
            λ = minimum(eigvals(full(T)))
            @show j, λ
            if b != nothing
                x = V[:, 1:j] * (T \ (V[:, 1:j]' * b))
                @show norm(A*x - b) / norm(b)
            end
        end
        # =======================================================

        if j < N
            w = w̃ - α[j] * V[:,j] - β[j] * V[:, j-1]
        end
    end
    β = β[2:end]
    return V, spdiagm((β, α, β), (-1,0,1))
end
