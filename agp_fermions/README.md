# AGP Fermi module

This is a super algebra encompassing the bare fermionic, pairing and spin-flip SU2 algebras, with their inter- and intra-algebraic commutation relations. As can always be done, one can also use this module to translate the pairing and the spin-flip operators into their fermionic representation.

## Usage guidelines and a brief summary of functions

1. `extract_su2`: Given a mixed expression containing fermionic creation / annihilation operators, this function identifies and converts the obvious SU2 operators. In the view of our current focus on the pairing algebra, spin-flip operators are left in their fermionic forms.

2. `spin_flip_to_fermi`:  Converts the spin-flip operators from SU2 to fermionic representation

3. `get_seniority_zero`:  Perhaps the most important function, it not only identifies the obvious SU2 operators from a string of mixed operators, but also forms all possible Kronecker Delta's between indices of a general fermion operator string to extract the seniority zero contribution.

4. `unique_indices`:  This function can be used to declare sets of indices to be unique. This functionality can only be used for free indices, that is those which are not being summed over. **The `unique_indices` function comes with most of the troubles and needs to be used with caution (see example below)**.

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

# 6. The caution with unique-indices
# THE INCORRECT / DANGEROUS WAY
# NOTE: Generally, we aren't going to be the ones to define this `expression`
#       but this will emerge as a result of some previous calculation
#       and may contain Kronecker Deltas.
Z11 = IndexedBase('Z11')
expression = h[p, s] * t[p, q, r, s] * Z11[s] * KroneckerDelta(q, r)
res = dr.sum(
    (p, A), (q, A), (r, A), (s, A),
    expression
)
res = res.simplify()

# Ideally this expression should be non-zero because the Kronecker Delta
# between q and r is allowed. But when drudge sees dummy indices, it likes to
# bring them into a systematic order first. In doing so, it relabels them in
# such a way as to end up with a KroneckerDelta(p, q).
# And then, unique indices demands this expression become zero.

# THE CORRECT WAY:
# Is to get rid of the declaration that the (p, q) and (r, s) indices are
# unique. All the info / functionality of the uniqueness of the indices
# has already been used and is no longer needed.
dr.purge_unique_indices()
res = dr.sum(
    (p, A), (q, A), (r, A), (s, A),
    expression
)
res = res.simplify()
```
