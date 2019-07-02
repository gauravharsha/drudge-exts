# Extensions to Drudge

This repository contains drudge modules which are not general enough to be added to the original software but very useful in our group.

1. `bcs_allindex`: The `bcs.py` module is complimentary to the existing ReducedBCSDrudge class. In this module, we prefer to work with general orbital indices rather than particle-hole indices. Furthermore, we also have a new function `eval_agp` that can be used to evaluate the expectation value of strings of Pdag, N, P over AGP (or any arbitrary) wavefunction.
