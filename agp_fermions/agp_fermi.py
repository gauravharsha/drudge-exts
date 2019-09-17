"""
Drudge for BCS-AGP-fermionic algebra
"""
import collections, functools, operator, re
import pdb

from sympy import Integer, Symbol, IndexedBase, KroneckerDelta, factorial, Function, srepr
from sympy.utilities.iterables import default_sort_key

from drudge import Tensor
from drudge.canon import IDENT,NEG
from drudge.fock import PartHoleDrudge, SpinOneHalfPartHoleDrudge
from drudge.canonpy import Perm
from drudge.term import Vec, Range, Term
from drudge.utils import sympy_key
from drudge.fock import SpinOneHalfGenDrudge

from bcs import *

class AGPFermi(SpinOneHalfGenDrudge):
    r"""
    Drudge module that deals primarily with fermions, but provides a
    functionality to evaluate expectation values over AGP wavefunction. In the
    process, it is convenient to have the following set of operators in the
    drudge module:
        :math:  `P^\dagger_p = c_{p,\uparrow}^\dagger c_{p,\downarrow}^\dagger`
        :math:  `N_p = n_{p,\uparrow} + n_{p,\downarrow}`
        :math:  `P_p = c_{p,\downarrow} c_{p,\uparrow}`

        :math:  `S^+_p = c_{p,\uparrow}^\dagger c_{p,\downarrow}`
        :math:  `S^z_p = \frac{1}{2}(n_{p,\uparrow} - n_{p,\downarrow})`
        :math:  `S^-_p = c_{p,\downarrow}^\dagger c_{p,\uparrow}`

    along with the spin-one-half fermion creation//annihilation operators
    """

    def __init__(
        self, ctx,
        all_orb_range=Range('A', 0, Symbol('norb')),
        all_orb_dumms=PartHoleDrudge.DEFAULT_ORB_DUMMS,
        **kwargs
    ):
        # Define super with the described orbital ranges
        orb = ((all_orb_range, all_orb_dumms),)
        super().__init__(ctx, orb=orb, **kwargs)

        # set the dummies
        self.set_dumms(all_orb_range, all_orb_dumms)
        self.add_resolver({
            i: (all_orb_range) for i in all_orb_dumms
        })

        #Pairing operators
        bcs_dr = ReducedBCSDrudge(ctx,
            all_orb_range=all_orb_range, all_orb_dumms=all_orb_dumms,
        )
        N_ = bcs_dr.cartan
        Pdag_ = bcs_dr.raise_
        P_ = bcs_dr.lower

        # SU2 operators
        su2_dr = SU2LatticeDrudge(ctx)
        Sz_ = su2_dr.cartan
        Sp_ = su2_dr.raise_
        Sm_ = su2_dr.lower

        # Assign these operators to the self
        self.all_orb_range = all_orb_range
        self.all_orb_dumms = all_orb_dumms
        self.Pdag = Pdag_
        self.N = N_
        self.P = P_
        self.S_z = Sz_
        self.S_p = Sp_
        self.S_m = Sm_

        # Set the names
        self.set_name(*self.all_orb_dumms)
        self.set_name(**{
            Sz_.label[0]+'_z' : Sz_,
            Sp_.label[0]+'_p' : Sp_,
            Sm_.label[0]+'_m' : Sm_,
            N_.label[0] : N_,
            N_.label[0]+'_' : N_,
            Pdag_.label[0]+'_p' : Pdag_,
            Pdag_.label[0]+'_dag' : Pdag_,
            P_.label[0]+'_m' : P_,
            P_.label[0]+'_' : P_,
        })


    # def normal_order(self, terms, **kwargs):
        # NOTE Define the functionality of normal / canonical order for the complicated
        # set of operators that we have -- we will invoke the existing normal_order
        # functions from the respective drudges and define relations for cross-
        # commutators.
