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



println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
println(" Starting `SaddleSearch` tests")
println("≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡")
@testset "SaddleSearch" begin
   @testset "StaticDimer vs BBDimer" begin include("test_staticvsbb.jl") end
   @testset "Vacancy" begin include("test_vacancy.jl") end
   @testset "LineSearchDimer" begin include("test_lsdimer.jl") end
end

