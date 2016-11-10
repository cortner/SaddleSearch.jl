

abstract DimerLineSearch


type StaticDimerLineSearch <: DimerLineSearch
   a_trans::Float64
   a_rot::Float64
end



dimermethod(E, dE, x0, a_trans::Float64, a_rot::Float64) =
   dimermethod(E, dE, x0, StaticDimerLineSearch(a_trans, a_rot))

"""
most basic variant of the dimer method, with alternating translation and
rotation steps.
"""
function dimermethod{T}(E, dE, x0::Vector{T},
                        linesearch::DimerLineSearch)
   error("to be implemented")
end


function lbfgsdimer{T}(E, dE, x0::Vector{T},
                        linesearch::DimerLineSearch)
   error("to be implemented")
end


function seqmindimer{T}(E, dE, x0::Vector{T})
   error("to be implemented")
end
