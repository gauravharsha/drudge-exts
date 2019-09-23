"""
Drudge for BCS-AGP-fermionic algebra
"""
import collections, functools, operator, re, typing

from sympy import Integer, Symbol, IndexedBase, KroneckerDelta, factorial, Function, srepr
from sympy.utilities.iterables import default_sort_key

from drudge import Tensor, TensorDef
from drudge.canon import IDENT,NEG
from drudge.genquad import GenQuadDrudge
from drudge.fock import SpinOneHalf, CranChar, PartHoleDrudge, SpinOneHalfPartHoleDrudge
from drudge.canonpy import Perm
from drudge.term import Vec, Range, Term
from drudge.utils import sympy_key
from drudge.fock import SpinOneHalfGenDrudge

from bcs import *

UP = SpinOneHalf.UP
DOWN = SpinOneHalf.DOWN

class AGPFermi(GenQuadDrudge):
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

    PAIRING_CARTAN = Vec(r'N')
    PAIRING_RAISE = Vec(r'P^\dagger')
    PAIRING_LOWER = Vec(r'P')

    SPIN_CARTAN = Vec(r'J^z')
    SPIN_RAISE = Vec(r'J^+')
    SPIN_LOWER = Vec(r'J^-')

    def __init__(
        self, ctx, op_label='c',
        all_orb_range=Range('A', 0, Symbol('norb')),
        all_orb_dumms=PartHoleDrudge.DEFAULT_ORB_DUMMS,
        spin_range=Range(r'\uparrow \downarrow', Integer(0), Integer(2)),
        spin_dumms=tuple(Symbol('sigma{}'.format(i)) for i in range(50)),
        bcs_N=PAIRING_CARTAN, bcs_Pdag=PAIRING_RAISE, bcs_P=PAIRING_LOWER,
        su2_Jz=SPIN_CARTAN, su2_Jp=SPIN_RAISE, su2_Jm=SPIN_LOWER,
        bcs_root=Integer(2), bcs_norm=Integer(1), bcs_shift=Integer(-1),
        su2_root=Integer(2), su2_norm=Integer(1), su2_shift=Integer(0),
        **kwargs
    ):

        # initialize super
        super().__init__(ctx, **kwargs)

        # Initialize SpinOneHalfGenDrudge with the described orbital ranges
        orb = ((all_orb_range, all_orb_dumms),(spin_range, spin_dumms))
        fermi_dr = SpinOneHalfGenDrudge(
            ctx, orb=orb, op_label=op_label, **kwargs
        )
        self.fermi_dr = fermi_dr

        cr = fermi_dr.cr
        an = fermi_dr.an
        self.cr = cr
        self.an = an

        # set the dummies
        self.set_dumms(all_orb_range, all_orb_dumms)
        self.set_dumms(spin_range, spin_dumms)

        # Add resolver for all orbital dummies
        self.add_resolver({
            i: (all_orb_range) for i in all_orb_dumms
        })

        # Define and add the spin range and dummy indices to the drudge module
        # XXX: Note that the spin dummies are useless in this module and must be
        #   removed eventually
        self.add_resolver({
            UP: spin_range,
            DOWN: spin_range
        })
        self.spin_range = spin_range
        self.spin_dumms = self.dumms.value[spin_range]

        #Pairing operators
        bcs_dr = ReducedBCSDrudge(ctx,
            all_orb_range=all_orb_range, all_orb_dumms=all_orb_dumms,
            cartan=bcs_N, raise_=bcs_Pdag, lower=bcs_P,
        )
        self.bcs_dr = bcs_dr
        N_ = bcs_dr.cartan
        Pdag_ = bcs_dr.raise_
        P_ = bcs_dr.lower

        # SU2 operators
        su2_dr = SU2LatticeDrudge(ctx,
            cartan=su2_Jz, raise_=su2_Jp, lower=su2_Jm,
        )
        self.su2_dr = su2_dr
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

        # Define the unitary group operators
        p = Symbol('p')
        q = Symbol('q')
        sigma = self.dumms.value[spin_range][0]
        self.e_ = TensorDef(Vec('E'), (p, q),self.sum(
            self.cr[p, UP] * self.an[q, UP] + self.cr[p, DOWN] * self.an[q, DOWN]
            )
        )
        self.set_name(e_=self.e_)

        # Define the Dpq, Ddag_pq operators

        # Set the names
        self.set_name(*self.all_orb_dumms)
        self.set_name(**{
            op_label+'_' : an,
            op_label+'_dag' : cr,
            op_label+'dag_' : cr,
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

        spec = _AGPFSpec(
            c_=self.an, c_dag=self.cr, N=self.N, P=self.P, Pdag=self.Pdag,
            agproot=bcs_root, agpnorm=bcs_norm, agpshift=bcs_shift,
            S_p=self.S_p, S_z=self.S_z, S_m=self.S_m,
        )
        self._spec = spec
        self._swapper = functools.partial(_swap_agpf, spec=spec)

    @property
    def swapper(self) -> GenQuadDrudge.Swapper:
        """The swapper for the AGPF algebra -- invoked only when at least one
        of the two vectors is SU2 or BCS generator
        """
        return self._swapper

    def _latex_vec(self, vec):
        """Get the LaTeX form of operators. This needs over-writing because the
        fermionic expressions encode creation and annihilation as an index,
        while the SU2 operators have daggers / + defined in the symbol definition.
        """

        if ((vec.base==self.cr.base) or (vec.base==self.an.base)):
            return self.fermi_dr._latex_vec(vec)
        else:
            return super()._latex_vec(vec)

_AGPFSpec = collections.namedtuple('_AGPFSpec',[
    'c_',
    'c_dag',
    'N',
    'Pdag',
    'P',
    'agproot',
    'agpnorm',
    'agpshift',
    'S_p',
    'S_z',
    'S_m',
])

_P_DAG = 0
_N_ = 1
_P_ = 2
_S_P = 3
_S_Z = 4
_S_M = 5
_C_ = 6
_C_DAG = 7

def _parse_vec(vec, spec: _AGPFSpec):
    """Get the character, lattice indices, and indices keys of the vector.
    """
    base = vec.base
    indices = vec.indices
    if base == spec.c_.base:
        if vec.indices[0]==CranChar.AN:
            char = _C_
            indices = vec.indices[1:]
        elif vec.indices[0]==CranChar.CR:
            char = _C_DAG
            indices = vec.indices[1:]
        else:
            pass
    elif base == spec.N:
        char = _N_
    elif base == spec.Pdag:
        char = _P_DAG
    elif base == spec.P:
        char = _P_
    elif base == spec.S_p:
        char = _S_P
    elif base == spec.S_z:
        char = _S_Z
    elif base == spec.S_m:
        char = _S_M
    else:
        raise ValueError('Unexpected vector for AGPFermi algebra', vec)

    keys = tuple(sympy_key(i) for i in indices)

    return char, indices, keys

def _swap_agpf(vec1: Vec, vec2: Vec, depth=None, *, spec: _AGPFSpec):
    if depth is None:
        depth = 1

    char1, indice1, key1 = _parse_vec(vec1, spec)
    char2, indice2, key2 = _parse_vec(vec2, spec)

    if len(indice1) == len(indice2):
        delta = functools.reduce(operator.mul, (
            KroneckerDelta(i, j) for i, j in zip(indice1, indice2)
        ), _UNITY)
    else:
        delta = KroneckerDelta(indice1[0], indice2[0])

    agp_root = spec.agproot
    agp_norm = spec.agpnorm
    agp_shift = spec.agpshift

    if char1 == _P_DAG:
        if char2 == _P_DAG:
            if key1 < key2:
                return None
            elif key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    elif char1 == _N_:
        if char2 == _P_DAG:
            return _UNITY, agp_root * delta * spec.Pdag[indice1]
        elif char2 == _N_:
            if key1 < key2:
                return None
            elif key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    elif char1 == _P_:
        if char2 == _P_DAG:
            return _UNITY, - agp_norm * delta * (spec.N[indice1] + agp_shift)
        elif char2 == _N_:
            return _UNITY, agp_root * delta * spec.P[indice1]
        elif char2 == _P_:
            if key1 < key2:
                return None
            elif key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    else:
        return None
        # assert False

_UNITY = Integer(1)
_NOUGHT = Integer(0)
_NEGONE = Integer(-1)
_TWO = Integer(2)
