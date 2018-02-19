export serial, palindrome

"""
energies along a path in the String and NEB-type methods can be computed either
in a 'serial' manner or in a 'palindrome' manner:

'serial' : the energies of the images along the path are computed always in the same direction.
'palindrome' : the order of computing the energies of the images along a path is reversed after each iteration.

### Parameters:
* `ord` : order of traversing the images along the path
"""

@with_kw type serial
   direction = (M, nit) -> 1:M
end

@with_kw type palindrome
   direction = (M, nit) -> M-mod(nit,2)*(M-1):2*mod(nit,2)-1:M-mod(nit+1,2)*(M-1)
end
