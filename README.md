# Extension modules for drudge

This repository contains drudge modules which are not broad enough to be added to the original software but very useful for specific applications.

1.  `bcs_allindex`:     a module is complimentary to the existing ReducedBCSDrudge class. In this module, we prefer to work with general orbital indices rather than particle-hole indices. Furthermore, we also have a new function `eval_agp` that can be used to evaluate the expectation value of strings of Pdag, N, P over AGP (or any arbitrary) wavefunction.

2.  `ugagp`:            provides the unitary group generators and the one-body AGP killers (i.e. `Ddag` and `D` operators) along with a way to translate from one to another. Again, as all AGP based modules, there is no distinction of particle / hole indices. Instead we want to work with general orbital indices.

3.  `su4`:              provides the algebra for treating generators of SU4 algebra.

4.  `agp_fermi`:        a module that allows working with fermions as well as SU(2) pairing and spin-flip operators, and at the end of the day, extract the seniority zero components in the given string of fermion creation / annihilation operators.

3.  `misc_examples`:    Contains basic examples for `drudge`.
