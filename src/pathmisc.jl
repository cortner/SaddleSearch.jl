using Dierckx
# -------------- Vectors and Paths  -----------------

export Path

struct Path{T, NI}
   x::Vector{T}
   valNI::Type{Val{NI}}
end
Path(x::Vector) = Path(x,Val{length(x)})

# vectorise an object of type Path
function Base.vec(x::Vector{T}) where {T} 
   return cat(x...; dims = 1)
end

# convert an object of type Path to type Vector
function Base.convert(::Type{Path{T,NI}}, X::Vector{<: AbstractFloat}) where {T, NI}
   return [ X[(n-1)*(length(X)÷NI)+1 : n*(length(X)÷NI)] for n=1:NI]
end

export serial, palindrome


# ------------- Path Traversing ------------

"""
energy evaluations along a path in the String/NEB-type methods can be performed
in a 'serial' manner or in a 'palindrome' manner:

* `serial` : traverse path always in the same direction.
* `palindrome` : the order of computing the energy of the images along a path is reversed after each iteration.

### Parameters:
* `direction` : order of traversing the images along the path
"""
@with_kw struct serial
   direction = (M, nit) -> 1:M
end

@with_kw struct palindrome
   direction = (M, nit) -> M-mod(nit,2)*(M-1):2*mod(nit,2)-1:M-mod(nit+1,2)*(M-1)
end



# ------------- Reparametrisation of Paths ------------

"""
parametrisation functions of paths for String/NEB-type methods
"""
function parametrise!(dxds::Vector{T}, x::Vector{T}, ds::T; 
                      parametrisation=range(0.,1.,length = length(x))
                     ) where {T}
   # interpolate and reparametrise path

   param = [0; [sum(ds[1:i]) for i in 1:length(ds)]]
   param /= param[end]; param[end] = 1.

   # cubic splines interpolation
   S = [Spline1D(param, [x[i][j] for i=1:length(x)], w = ones(length(x)), k = 3, bc = "error") for j=1:length(x[1])]

   # reparametrise
   xref = [[Sj(s) for s in parametrisation] for Sj in S ]
   dxdsref = [[derivative(Sj, s) for s in parametrisation] for Sj in S]
   d²xds² = [[derivative(Sj, s, nu=2) for Sj in S] for s in parametrisation]

   # return parametrised variables
   x_ = cat(xref...; dims = 2)
   dxds_ = cat(dxdsref...; dims = 2)
   [x[i] = x_[i,:] for i=1:length(x)]
   [dxds[i] = dxds_[i,:] for i=1:length(x)]
   return d²xds²
end

# distribute images uniformly with respect to P
function redistribute(X::Vector{Float64}, path_type::Type{Path{T,NI}}, precon, precon_scheme) where {T,NI}

   x = convert(path_type, X)
   t = deepcopy(x)

   Np = length(precon);
   function P(i) return precon[mod(i-1,Np)+1, 1]; end
   function P(i, j) return precon[mod(i-1,Np)+1, mod(j-1,Np)+1]; end

   ds = [dist(precon_scheme, P, x, i) for i=1:length(x)-1]
   parametrise!(t, x, ds)

   return vec(x)
end



# -------------- Preconditioning for Paths  -----------------

export localPrecon, globalPrecon

"""
the two preconditioning schemes implemented for String/NEB-type methods

`localPrecon` : the preconditioner acts as a coordinate transformation of the
state space.
`globalPrecon` : the preconditioner is applied to the force directly, each point
along the path is preconditioned independently.

### Parameters:
* `precon` : preconditioner
* `precon_prep!` : update function for preconditioner
* `distance` : distance metric

### Functionality:
* `dist` : distance of neighbouring images along path
* `dot_P` : preconditioned dot product of associated images in two paths x, y
* `point_norm` : local P-norm
* `proj_grad` : projected gradient
* `forcing` : reference vector of preconditioned forces along path
* `elastic_force` : Hooke's law elastic focres along path
* `maxres` : residual error
"""

@with_kw struct localPrecon
   precon = I
   precon_prep! = (P, x) -> P
   distance = (P, x1, x2) -> norm(P, x2 - x1)
end

@with_kw struct globalPrecon
   precon = I
   precon_prep! = (P, x) -> P
   distance = (P, x1, x2) -> norm(x2 - x1)
end

dist(precon_scheme::localPrecon, P, x, i) = precon_scheme.distance(0.5*(P(i)+P(i+1)), x[i], x[i+1])
dist(precon_scheme::globalPrecon, P, x, i) = precon_scheme.distance(x[i], x[i+1])

dot_P(precon_scheme::localPrecon, x, P, y) = sum([dot(x[i], P(i), y[i]) for i=1:length(x)])
dot_P(precon_scheme::globalPrecon, x, P, y) = sum([dot(x[i], y[i]) for i=1:length(x)])

point_norm(precon_scheme::localPrecon, P, dxds) = [ 1; [norm(P(i), dxds[i])
                                                    for i=2:length(dxds)-1]; 1 ]
point_norm(precon_scheme::globalPrecon, P, dxds) = [norm(dxds[i]) for i=1:length(dxds)]

proj_grad(precon_scheme::localPrecon, P, ∇E, dxds) = -[P(i) \ ∇E[i] - dot(∇E[i],dxds[i])*dxds[i] for i=1:length(dxds)]
proj_grad(precon_scheme::globalPrecon, P, ∇E, dxds) = vec(-[∇E[i] - dot(∇E[i],dxds[i])*dxds[i] for i=1:length(dxds)])

forcing(precon_scheme::localPrecon, P, neg∇Eperp) = return vec(neg∇Eperp)
forcing(precon_scheme::globalPrecon, P, neg∇Eperp) = vec(P) \ neg∇Eperp

function elastic_force(precon_scheme::localPrecon, P, κ, dxds, d²xds²)
   zz = zeros(eltype(dxds[1]), length(dxds[1]))
    return - [ [copy(zz)];
               κ*[dot(d²xds²[i], P(i), dxds[i]) * dxds[i] for i=2:length(dxds)-1];
               [copy(zz)] ]
end
function elastic_force(precon_scheme::globalPrecon, P, κ, dxds, d²xds²)
   zz = zeros(eltype(dxds[1]), length(dxds[1]))
    return vec(-[ [copy(zz)];
               κ*[dot(d²xds²[i], dxds[i]) * dxds[i] for i=2:N-1];
               [copy(zz)] ])
end

maxres(precon_scheme::localPrecon, P, ∇Eperp) =  maximum([norm(P(i)*∇Eperp[i],Inf)
                                                for i = 1:length(∇Eperp)])
maxres(precon_scheme::globalPrecon, P, ∇Eperp) = norm(∇Eperp, Inf)
