using SaddleSearch, SaddleSearch.TestSets
using Base.Test

verbose=1

runtest(t) = include("test_$(t).jl")

tests = [
   "staticvsbb",
   "lsdimer",
   "superlinear",
   "vacancy",
]

println("Starting `SaddleSearch` tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
for test in tests
      runtest(test)
end


# println("Starting `SaddleSearch` Benchmarks")
# println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
# include("dimer_benchmarks.jl")
