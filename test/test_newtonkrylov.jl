@testset "newtonkrylov" begin

println("Testing `dcg1_index1`")
@testset "dcg1_index1" begin

d = 30
x = linspace(-1,1,d)
A = d^2 * SymTridiagonal(2*ones(d), -ones(d-1))
B = Diagonal(exp(- x.^2) + 0.1)

# construct an index-1 system
μ = 30.0
H = A - μ * B
σ = sort(eigvals(full(H)))
@assert σ[1] < 0 && σ[2] > 0

b = ones(d) + 0.1 * sin.(x)

# define a linear function
f = x -> H * x
x, λ, v, _ = dcg_index1(zeros(d), f, zeros(d), 1e-6, d; b = b, debug = false)
@test norm(H * x - b) < 1e-10
@test norm(H * v - λ * v, Inf) < 1e-10
@test abs(λ - σ[1]) < 1e-10

# now try the same with a quadratic nonlinearity mixed in
#   (hessian remains the same!)
q = x -> [ [x[i]*x[i+1] for i = 1:d-1]; x[end] * x[1] ]
f = x -> H * x + q(x)
x, λ, v, _ = dcg_index1(zeros(d), f, zeros(d), 1e-6, d; b = b, debug = false)
@test norm(H * x - b) < 1e-6
@test norm(H * v - λ * v, Inf) < 1e-7
@test abs(λ - σ[1]) < 1e-7

# next we add some preconditioning
P = 0.9 * sparse(A) + μ/2 * speye(d)
λP = minimum(eigvals(full(H), full(P)))
x, λ, v, numf = dcg_index1(zeros(d), f, zeros(d), 1e-6, d; P = P, b = b, debug = false)
# TODO: test that it took just 9 iterations!
@test numf == 9
@test norm(H * x - b) < 1e-6
@test norm(H * v - λ * P * v, Inf) < 1e-7
@test abs(λ - λP) < 1e-7
if numf != 9
   warn("numf was 9 in original tests; now it is $numf")
end

end  # @testset "dcg1_index1"



end  # @testset "newtonkrylov"
