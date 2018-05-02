using SaddleSearch, SaddleSearch.TestSets
using Base.Test
using SaddleSearch: numE, numdE, res_trans, res_rot

# Available Test Sets:
#   MullerPotential
#   DoubleWell
#   LJcluster
#   LJVacancy2D
#   Molecule2D

verbose=1

runtest(t) = include("test_$(t).jl")

tests = [
   "staticvsbb",
   "lsdimer",
   "superlinear",
   "vacancy",
   "molecule2d",
   # "newtonkrylov",
]

println("Starting `SaddleSearch` tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
@testset "SaddleSearch" begin
   for test in tests
         runtest(test)
   end
end
