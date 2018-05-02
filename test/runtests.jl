using SaddleSearch
using Base.Test
using SaddleSearch: numE, numdE, res_trans, res_rot, maxres
include("testsets.jl"); using TestSets

function heading1(str)
   dashes = "-"^length(str)
   print_with_color(:magenta, dashes); println()
   print_with_color(:magenta, str, bold=true); println()
   print_with_color(:magenta, dashes);println()
end

heading2(str) = print_with_color(:green, str*"\n", bold=true)


# change this to 2 to get full information or to 0 to get virtually no information
verbose=1

# test files are test_*** where *** is the string in this list
tests = [
   # "staticvsbb",
   # "lsdimer",
   "vacancy",
]


println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println(" Starting `SaddleSearch` tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
@testset "SaddleSearch" begin
   for test in tests
      include("test_$(test).jl")
   end
end
