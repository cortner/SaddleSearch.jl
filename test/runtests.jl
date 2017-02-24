using SaddleSearch, SaddleSearch.TestSets
using Base.Test
using SaddleSearch: numE, numdE, res_trans, res_rot

verbose=1

runtest(t) = include("test_$(t).jl")

tests = [
   "staticvsbb",
   "lsdimer",
   "superlinear",
   "vacancy",
   "molecule2d",
]

println("Starting `SaddleSearch` tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
for test in tests
      runtest(test)
end
