using SaddleSearch, SaddleSearch.TestSets
using Base.Test

verbose=1


# tests = [
#    "staticvsbb",
#    "lsdimer",
# ]
#
# println("Starting `SaddleSearch` tests")
# println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
# for test in tests
#    include("test_$(test).jl")
# end


# println("Starting `SaddleSearch` Benchmarks")
# println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
#
# include("dimer_benchmarks.jl")


include("test_superlinear.jl")
