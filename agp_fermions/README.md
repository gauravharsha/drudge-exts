# AGP Fermi module

This is a super algebra encompassing the bare fermionic, pairing and spin-flip SU2 algebras, with their inter- and intra-algebraic commutation relations. As can always be done, one can also use this module to translate the pairing and the spin-flip operators into their fermionic representation. There are two versions: the general index version, and the particle-hole extension. In the latter, all the sums over general indices (such as `p`, `q`, `r`, `s`, etc.) are broken down into a sum over occupied an virtual orbitals.

## Usage guidelines and a brief summary of functions

1. `extract_su2`: Given a mixed expression containing fermionic creation / annihilation operators, this function identifies and converts the obvious SU2 operators. In the view of our current focus on the pairing algebra, spin-flip operators are left in their fermionic forms.

2. `spin_flip_to_fermi`:  Converts the spin-flip operators from SU2 to fermionic representation

3. `get_seniority_zero`:  Perhaps the most important function, it not only identifies the obvious SU2 operators from a string of mixed operators, but also forms all possible Kronecker Delta's between indices of a general fermion operator string to extract the seniority zero contribution.

4. `unique_indices`:  This function can be used to declare sets of indices to be unique. Once an index is declared to be unique, it will be rmoved from the pool of dummy indices in order to avoid conflicts.

5. `unique_del_lists`:  Class attribute that contains information about the sets of indices that are declared to be unique.

## Example
In this example, we will explore the usage of some of the functions mentioned above and see the caution associated with `unique_indices`.

```python
from dummy_spark import SparkContext
from sympy import *
from drudge import *
from agp_fermi import *

# 1. Initialize drudge
ctx = SparkContext()
dr = AGPFermi(ctx)

# 2. Extract some of the commonly needed operators from the name space
#       a.  e_: Unitary group operator
#       b.  A:  Range for the all-orbital-dummies
#       c.  all_orb_dumms:  Dummy indices associated with names.A
names = dr.names
e_ = dr.e_
A = names.A
p, q, r, s, x, y = dr.all_orb_dumms[:6]

# 3. Let's define a one-body Hamiltonian
h = IndexedBase('h')
dr.set_symm(h, Perm([1, 0], IDENT))
ham1 = dr.simplify(dr.einst(h[p, q] * e_[p, q]))

# 4. Define the 4 index tensor
t = IndexedBase('t')
dr.set_symm(
    t,
    Perm([1, 0, 2, 3], NEG),
    Perm([0, 1, 3, 2], NEG),
    Perm([2, 3, 0, 1], IDENT)
)

# 5. Set the unique indices
dr.unique_indices([p, q])
dr.unique_indices([r, s])

print(dr.unique_del_lists)

# 6. Perform some calculation
Z11 = IndexedBase('Z11')
expression = h[p, s] * t[p, q, r, s] * Z11[s] * KroneckerDelta(q, r)
res = dr.sum(
    (p, A), (q, A), (r, A), (s, A),
    expression
)
res = res.simplify()
```
