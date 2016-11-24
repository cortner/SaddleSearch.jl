using SaddleSearch, SaddleSearch.TestSets
using Base.Test

verbose=1

# import the different test sets
# include("testsets.jl")

tests = [
   "staticvsbb",
   "lsdimer",
]

println("Starting `SaddleSearch` tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
for test in tests
   include("test_$(test).jl")
end


println("Starting `SaddleSearch` Benchmarks")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")

include("dimer_benchmarks.jl")
