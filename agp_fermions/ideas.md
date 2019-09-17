# Module to carry out AGP algebra using fermionic drudge

The UGA-AGP or the AGP-UGA drudge modules are inefficient for the following reasons:

    1. If we choose to work with the `Ddag` and `D` operators, the commutation rules are very complicated.

    2. If we chooe to work with the `E_pq` operators, while the commutation rules are simplified, one eventually has to transform `E`'s back to `D`'s and carry out a normal-ordering there to get obtain the final expectation values.

To overcome these difficulties, we need a new module with the following feature:

    1. Supports the following algebras -SU(2) pairing and fermionic anti-commutation algebra.
    2. Encodes commutation rules between the Pairing and the fermion operators.
    3. Has a function that can evaluate the expectation values of a string of fermions over the AGP wavefunction. 
