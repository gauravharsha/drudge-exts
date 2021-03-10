"""
Drudge for BCS-AGP-fermionic algebra
"""
import collections
import functools
import operator

from sympy import (
    Symbol, KroneckerDelta, Integer, Mul, Number, Expr, Rational, ordered
)

from drudge import Tensor, TensorDef
from drudge.genquad import GenQuadDrudge
from drudge.fock import SpinOneHalf, CranChar
from drudge.term import Vec, Range, Term, try_resolve_range
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

    NUMBER_UP = Vec(r'n^{\uparrow}')
    NUMBER_DN = Vec(r'n^{\downarrow}')

    SPIN_CARTAN = Vec(r'J^z')
    SPIN_RAISE = Vec(r'J^+')
    SPIN_LOWER = Vec(r'J^-')

    DEFAULT_ORB_DUMMS = tuple(Symbol(i) for i in 'pqrs') + tuple(
        Symbol('p{}'.format(i)) for i in range(50)
    )

    def __init__(
        self, ctx, op_label='c',
        all_orb_range=Range('A', 0, Symbol(r'M_orb')),
        all_orb_dumms=DEFAULT_ORB_DUMMS,
        spin_range=Range(r'\uparrow \downarrow', Integer(0), Integer(2)),
        spin_dumms=tuple(Symbol('sigma{}'.format(i)) for i in range(50)),
        bcs_N=PAIRING_CARTAN, bcs_Pdag=PAIRING_RAISE, bcs_P=PAIRING_LOWER,
        bcs_Nup=NUMBER_UP, bcs_Ndn=NUMBER_DN,
        su2_Jz=SPIN_CARTAN, su2_Jp=SPIN_RAISE, su2_Jm=SPIN_LOWER,
        bcs_root=Integer(2), bcs_norm=Integer(1), bcs_shift=Integer(-1),
        su2_root=Integer(1), su2_norm=Integer(2), su2_shift=Integer(0),
        **kwargs
    ):

        # initialize super
        super().__init__(ctx, **kwargs)

        # Initialize SpinOneHalfGenDrudge with the described orbital ranges
        orb = ((all_orb_range, all_orb_dumms), (spin_range, spin_dumms))
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
        # XXX: Note that the spin dummies are useless in this module and must
        #   be removed eventually
        self.add_resolver({
            UP: spin_range,
            DOWN: spin_range
        })
        self.spin_range = spin_range
        self.spin_dumms = self.dumms.value[spin_range]

        # Pairing operators
        bcs_dr = ReducedBCSDrudge(
            ctx,
            all_orb_range=all_orb_range, all_orb_dumms=all_orb_dumms,
            cartan=bcs_N, raise_=bcs_Pdag, lower=bcs_P,
        )
        self.bcs_dr = bcs_dr
        N_ = bcs_dr.cartan
        Pdag_ = bcs_dr.raise_
        P_ = bcs_dr.lower
        self.eval_agp = bcs_dr.eval_agp

        # SU2 operators
        su2_dr = SU2LatticeDrudge(
            ctx,
            cartan=su2_Jz, raise_=su2_Jp, lower=su2_Jm,
        )
        self.su2_dr = su2_dr
        Sz_ = su2_dr.cartan
        Sp_ = su2_dr.raise_
        Sm_ = su2_dr.lower

        # Assign these operators to the self
        self.all_orb_range = all_orb_range
        self.all_orb_dumms = tuple(all_orb_dumms)
        self.Pdag = Pdag_
        self.N = N_
        self.N_up = bcs_Nup
        self.N_dn = bcs_Ndn
        self.P = P_
        self.S_z = Sz_
        self.S_p = Sp_
        self.S_m = Sm_

        # Define the unitary group operators
        p = Symbol('p')
        q = Symbol('q')
        self.sigma = self.dumms.value[spin_range][0]
        self.e_ = TensorDef(Vec('E'), (p, q), self.sum(
            self.cr[p, UP] * self.an[q, UP]
            + self.cr[p, DOWN] * self.an[q, DOWN]
            )
        )
        self.set_name(e_=self.e_)

        # set of unique dummies:
        # The idea is to declare a set of (free) dummy indices to be unique,
        # i.e. they have unique, different values by construction.
        # This is a feature of this module / class but has potential to be a
        # part of the drudge system.
        # The way I want to implement this is as follows:
        #   1. User specifies a tuple/list of indices to be set unique
        #   2. Then we construct a dictionary of all proosible kronecker deltas
        #       which will be zero
        #   3. in simplify / get_seniority_zero, we use this substitution.

        # Dictionary of substitutions
        self.unique_del_lists = []

        # XXX: To be doen / can be done
        # Define the Dpq, Ddag_pq operators

        # Set the names
        self.set_name(*self.all_orb_dumms)
        self.set_name(**{
            op_label+'_': an,
            op_label+'_dag': cr,
            op_label+'dag_': cr,
            Sz_.label[0]+'_z': Sz_,
            Sp_.label[0]+'_p': Sp_,
            Sm_.label[0]+'_m': Sm_,
            N_.label[0]: N_,
            N_.label[0]+'_': N_,
            Pdag_.label[0]+'dag': Pdag_,
            Pdag_.label[0]+'_dag': Pdag_,
            P_.label[0]: P_,
            P_.label[0]+'_': P_,
        })

        # Define spec for all the class methods needed for
        # extracting the su2 operators
        spec = _AGPFSpec(
            c_=self.an, c_dag=self.cr, N=self.N, Nup=self.N_up, Ndn=self.N_dn,
            P=self.P, Pdag=self.Pdag, agproot=bcs_root, agpnorm=bcs_norm,
            agpshift=bcs_shift, S_p=self.S_p, S_z=self.S_z, S_m=self.S_m,
            su2root=su2_root, su2norm=su2_norm, su2shift=su2_shift,
            unique_ind=self.unique_del_lists
        )
        self._spec = spec

        # Swapper dummy function for commutation rules
        self._swapper = functools.partial(_swap_agpf, spec=spec)

        # Extracting SU2 dummy function
        self._extract_su2 = functools.partial(_get_su2_vecs, spec=spec)

    # Do not use `\otimes' in latex expressions for the operators.
    _latex_vec_mul = ' '

    @property
    def swapper(self) -> GenQuadDrudge.Swapper:
        """The swapper for the AGPF algebra -- invoked only when at least one
        of the two vectors is SU2 or BCS generator
        """
        return self._swapper

    def _latex_vec(self, vec):
        """Get the LaTeX form of operators. This needs over-writing because the
        fermionic expressions encode creation and annihilation as an index,
        while the SU2 operators have daggers or + defined in the symbol
        definition.
        """

        if ((vec.base == self.cr.base) or (vec.base == self.an.base)):
            return self.fermi_dr._latex_vec(vec)
        else:
            return super()._latex_vec(vec)

    def normal_order(self, terms, **kwargs):
        """Normal ordering sequence for algebra
        """

        noed = super().normal_order(terms, **kwargs)
        noed = noed.filter(_nonzero_by_nilp)
        noed = noed.filter(_nonzero_by_cartan)

        noed = noed.flatMap(
            functools.partial(
                _canonicalize_indices, spec=self._spec
            )
        )

        return noed

    def unique_indices(self, indlist):
        """
        Function that takes a list / tuple of indices, which would be unique
        among themselves, and then update the dictionary of substitutions
        """
        # Extract the unique set of indices
        unq_ind = set(indlist)

        # Update the list
        self.unique_del_lists.append(unq_ind)

        # Remove the indices from the dummy indices
        for ind in indlist:
            orb_range = try_resolve_range(ind, {}, self.resolvers.value)
            if orb_range is None:
                continue

            if not isinstance(orb_range, collections.Iterable):
                orb_range = (orb_range, )

            if all(
                [ind not in self._dumms.var[rg] for rg in orb_range]
            ):
                continue
            else:
                for rg in tuple(orb_range):
                    self._dumms.var[rg].remove(ind)

        return

    def purge_unique_indices(self):
        """
        Reset the unique_del_substs dictionary to empty
        """

        self.unique_del_lists.clear()

        # Reset the dummy values
        self.set_dumms(self.all_orb_range, self.all_orb_dumms)

        return

    def canon_indices(self, expression: Tensor):
        """Bind function to canonicalize free / external indices.
        """
        return expression.bind(
            functools.partial(
                _canonicalize_indices, spec=self._spec
            )
        )

    def extract_su2(self, expression: Tensor):
        """Bind function to map fermion strings to obvious SU2 (Pairing as well
        as spin-flip operators)
        """
        return expression.bind(self._extract_su2)

    def spin_flip_to_fermi(self, tnsr: Tensor):
        """Substitute all the Spin flip operators with their respective fermionic
        strings"""

        gen_idx = self.all_orb_dumms[0]
        sp_def = self.define(
            SPIN_RAISE, gen_idx,
            cr[gen_idx, SpinOneHalf.UP]*an[gen_idx, SpinOneHalf.DOWN]
        )
        sm_def = self.define(
            SPIN_LOWER, gen_idx,
            cr[gen_idx, SpinOneHalf.DOWN]*an[gen_idx, SpinOneHalf.UP]
        )
        spin_defs = [sp_def, sm_def]

        return Tensor(
            self, tnsr.subst_all(spin_defs).terms
        )

    def get_seniority_zero(self, tnsr: Tensor):
        """Get the seniority zero component of the given operator expression.
        """

        # 1, simplify -- includes canonicalization
        expr1 = self.simplify(self.simplify(tnsr))

        # 2, throw away terms with odd number of fermion terms
        expr = self.simplify(
            expr1.filter(
                functools.partial(_even_fermi_filter, spec=self._spec)
            )
        )

        # 3, extract su2 terms
        expr1 = self.simplify(
            self.simplify(
                self.extract_su2(expr)
            )
        )

        # 4, get partitions
        expr = self.simplify(
            self.simplify(
                expr1.bind(
                    functools.partial(_get_fermi_partitions, spec=self._spec)
                )
            )
        )

        # 5, Follow steps 0, 1 and 3, a few times
        expr2 = self.simplify(expr)

        for i in range(10):
            expr1 = expr2*Integer(1)
            expr2 = self.simplify(
                self.extract_su2(expr1)
            )
            if self.simplify(expr - expr2) == 0:
                break

        # 6, Anything remaining with cdag, c --> we drop
        expr = self.simplify(
            expr2.filter(
                functools.partial(_no_fermi_filter, spec=self._spec)
            )
        )

        # 7. Substitute N_up and N_dn with N/2
        p = self.all_orb_dumms[0]
        expr2 = self.simplify(
            expr.subst(self.N_up[p], self.N[p]/Integer(2))
        )
        expr = self.simplify(
            expr2.subst(self.N_dn[p], self.N[p]/Integer(2))
        )

        return expr


class PartHoleAGPFermi(AGPFermi):
    """
    Particle-hole variation of the AGP Fermi module.
    """

    PAIRING_CARTAN = Vec(r'N')
    PAIRING_RAISE = Vec(r'P^\dagger')
    PAIRING_LOWER = Vec(r'P')

    NUMBER_UP = Vec(r'n^{\uparrow}')
    NUMBER_DN = Vec(r'n^{\downarrow}')

    SPIN_CARTAN = Vec(r'J^z')
    SPIN_RAISE = Vec(r'J^+')
    SPIN_LOWER = Vec(r'J^-')

    DEFAULT_ORB_DUMMS = tuple(Symbol(i) for i in 'pqrs') + tuple(
        Symbol('p{}'.format(i)) for i in range(50)
    )

    DEFAULT_PART_DUMMS = tuple(Symbol(i) for i in 'ijkl') + tuple(
        Symbol('i{}'.format(i)) for i in range(50)
    )

    DEFAULT_HOLE_DUMMS = tuple(Symbol(i) for i in 'abcd') + tuple(
        Symbol('a{}'.format(i)) for i in range(50)
    )

    def __init__(
        self, ctx, op_label='c',
        all_orb_range=Range('A', 0, Symbol(r'M')),
        all_orb_dumms=DEFAULT_ORB_DUMMS,
        part_range=Range('O', 0, Symbol('no')), part_dumms=DEFAULT_PART_DUMMS,
        hole_range=Range('V', 0, Symbol('nv')), hole_dumms=DEFAULT_HOLE_DUMMS,
        spin_range=Range(r'\uparrow \downarrow', Integer(0), Integer(2)),
        spin_dumms=tuple(Symbol('sigma{}'.format(i)) for i in range(50)),
        bcs_N=PAIRING_CARTAN, bcs_Pdag=PAIRING_RAISE, bcs_P=PAIRING_LOWER,
        bcs_Nup=NUMBER_UP, bcs_Ndn=NUMBER_DN,
        su2_Jz=SPIN_CARTAN, su2_Jp=SPIN_RAISE, su2_Jm=SPIN_LOWER,
        bcs_root=Integer(2), bcs_norm=Integer(1), bcs_shift=Integer(-1),
        su2_root=Integer(1), su2_norm=Integer(2), su2_shift=Integer(0),
        **kwargs
    ):

        # Initialize super
        super().__init__(
            ctx, op_label=op_label, all_orb_range=all_orb_range,
            all_orb_dumms=all_orb_dumms, spin_range=spin_range,
            spin_dumms=spin_dumms, bcs_N=bcs_N, bcs_Pdag=bcs_Pdag, bcs_P=bcs_P,
            bcs_Nup=bcs_Nup, bcs_Ndn=bcs_Ndn, su2_Jz=su2_Jz, su2_Jp=su2_Jp,
            su2_Jm=su2_Jm, bcs_root=bcs_root, bcs_norm=bcs_norm,
            bcs_shift=bcs_shift, su2_root=su2_root, su2_norm=su2_norm,
            su2_shift=su2_shift, **kwargs
        )

        # Add the part-hole indices and ranges to the class varables
        self.part_dumms = tuple(part_dumms)
        self.hole_dumms = tuple(hole_dumms)
        self.part_range = part_range
        self.hole_range = hole_range

        # Add the indices to the name space
        self.set_name(*self.part_dumms)

        # Link the dummy indices to their respective ranges
        self.set_dumms(self.part_range, self.part_dumms)
        self.set_dumms(self.hole_range, self.hole_dumms)

        # Clean up the default resolver
        self._resolvers.var.clear()

        # Add the resolver
        self.add_resolver({
            i: (self.part_range) for i in self.part_dumms
        })
        self.add_resolver({
            i: (self.hole_range) for i in self.hole_dumms
        })
        self.add_resolver({
            i: (self.part_range, self.hole_range) for i in self.all_orb_dumms
        })
        self.add_resolver({
            i: (self.all_orb_range) for i in self.all_orb_dumms
        })

    def purge_unique_indices(self):
        """
        Reset the unique_del_substs dictionary to empty
        """

        self.unique_del_lists.clear()

        # Reset the dummy values
        self.set_dumms(self.all_orb_range, self.all_orb_dumms)
        self.set_dumms(self.part_range, self.part_dumms)
        self.set_dumms(self.hole_range, self.hole_dumms)

        return


_AGPFSpec = collections.namedtuple('_AGPFSpec', [
    'c_',
    'c_dag',
    'N',
    'Nup',
    'Ndn',
    'Pdag',
    'P',
    'agproot',
    'agpnorm',
    'agpshift',
    'S_p',
    'S_z',
    'S_m',
    'su2root',
    'su2norm',
    'su2shift',
    'unique_ind',
])

_P_DAG = 0
_N_UP_ = 1
_N_DN_ = 2
_N_ = 3
_P_ = 4
_S_P = 5
_S_Z = 6
_S_M = 7
_C_ = 8
_C_DAG = 9


def _parse_vec(vec, spec: _AGPFSpec):
    """Get the character, lattice indices, and indices keys of the vector.
    """
    base = vec.base
    indices = vec.indices
    if base == spec.c_.base:
        if vec.indices[0] == CranChar.AN:
            char = _C_
            indices = vec.indices[1:]
        elif vec.indices[0] == CranChar.CR:
            char = _C_DAG
            indices = vec.indices[1:]
        else:
            pass
    elif base == spec.Nup:
        char = _N_UP_
    elif base == spec.Ndn:
        char = _N_DN_
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

    keys = tuple(sympy_key(i) for i in indices[0:1])

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

    su2_root = spec.su2root
    su2_norm = spec.su2norm
    su2_shift = spec.su2shift

    if char1 == _P_DAG:
        if char2 == _P_DAG:
            if key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    elif char1 == _N_UP_:
        if char2 == _P_DAG:
            return _UNITY, agp_root * delta * spec.Pdag[indice1] / Integer(2)
        elif char2 == _N_UP_:
            if key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    elif char1 == _N_DN_:
        if char2 == _P_DAG:
            return _UNITY, agp_root * delta * spec.Pdag[indice1] / Integer(2)
        elif char2 == _N_UP_:
            return _UNITY, _NOUGHT
        elif char2 == _N_DN_:
            if key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    elif char1 == _N_:
        if char2 == _P_DAG:
            return _UNITY, agp_root * delta * spec.Pdag[indice1]
        elif char2 == _N_UP_:
            return _UNITY, _NOUGHT
        elif char2 == _N_DN_:
            return _UNITY, _NOUGHT
        elif char2 == _N_:
            if key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    elif char1 == _P_:
        if char2 == _P_DAG:
            return _UNITY, - agp_norm * delta * (spec.N[indice1] + agp_shift)
        elif char2 == _N_UP_:
            return _UNITY, agp_root * delta * spec.P[indice1] / Integer(2)
        elif char2 == _N_DN_:
            return _UNITY, agp_root * delta * spec.P[indice1] / Integer(2)
        elif char2 == _N_:
            return _UNITY, agp_root * delta * spec.P[indice1]
        elif char2 == _P_:
            if key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    elif char1 == _S_P:
        if char2 == _P_DAG:
            return _UNITY, _NOUGHT
        elif char2 == _N_UP_:
            return _UNITY, _NEGONE * delta * spec.S_p[indice1]
        elif char2 == _N_DN_:
            return _UNITY, delta * spec.S_p[indice1]
        elif char2 == _N_:
            return _UNITY, _NOUGHT
        elif char2 == _P_:
            return _UNITY, _NOUGHT
        elif char2 == _S_P:
            if key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    elif char1 == _S_Z:
        if char2 == _P_DAG:
            return _UNITY, _NOUGHT
        elif char2 == _N_UP_:
            return _UNITY, _NOUGHT
        elif char2 == _N_DN_:
            return _UNITY, _NOUGHT
        elif char2 == _N_:
            return _UNITY, _NOUGHT
        elif char2 == _P_:
            return _UNITY, _NOUGHT
        elif char2 == _S_P:
            return _UNITY, su2_root * delta * spec.S_p[indice1]
        elif char2 == _S_Z:
            if key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    elif char1 == _S_M:
        if char2 == _P_DAG:
            return _UNITY, _NOUGHT
        elif char2 == _N_UP_:
            return _UNITY, delta * spec.S_m[indice1]
        elif char2 == _N_DN_:
            return _UNITY, _NEGONE * delta * spec.S_m[indice1]
        elif char2 == _N_:
            return _UNITY, _NOUGHT
        elif char2 == _P_:
            return _UNITY, _NOUGHT
        elif char2 == _S_P:
            return _UNITY, - su2_norm * delta * (spec.S_z[indice1] + su2_shift)
        elif char2 == _S_Z:
            return _UNITY, su2_root * delta * spec.S_m[indice1]
        elif char2 == _S_M:
            if key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    elif char1 == _C_DAG:
        if char2 == _P_DAG:
            return _UNITY, _NOUGHT
        elif char2 == _N_UP_:
            if indice1[1] == SpinOneHalf.UP:
                return _UNITY, _NEGONE * delta * spec.c_dag[indice1]
            elif indice1[1] == SpinOneHalf.DOWN:
                return _UNITY, _NOUGHT
            else:
                assert False
        elif char2 == _N_DN_:
            if indice1[1] == SpinOneHalf.DOWN:
                return _UNITY, _NEGONE * delta * spec.c_dag[indice1]
            elif indice1[1] == SpinOneHalf.UP:
                return _UNITY, _NOUGHT
            else:
                assert False
        elif char2 == _N_:
            return _UNITY, _NEGONE * delta * spec.c_dag[indice1]
        elif char2 == _P_:
            if indice1[1] == SpinOneHalf.UP:
                return _UNITY, \
                    _NEGONE * delta * spec.c_[indice1[0], SpinOneHalf.DOWN]
            elif indice1[1] == SpinOneHalf.DOWN:
                return _UNITY, delta * spec.c_[indice1[0], SpinOneHalf.UP]
            else:
                assert False
        elif char2 == _S_P:
            return _UNITY, _NOUGHT
        elif char2 == _S_Z:
            return _UNITY, _NEGHALF * delta * spec.c_dag[indice1]
        elif char2 == _S_M:
            if indice1[1] == SpinOneHalf.UP:
                return _UNITY, \
                    _NEGONE * delta * spec.c_dag[indice1[0], SpinOneHalf.DOWN]
            elif indice1[1] == SpinOneHalf.DOWN:
                return _UNITY, \
                    _NEGONE * delta * spec.c_dag[indice1[0], SpinOneHalf.UP]
            else:
                assert False
        elif char2 == _C_DAG:
            if key1 > key2:
                return _NEGONE, _NOUGHT
            elif key1 == key2:
                if indice2[1] == SpinOneHalf.UP:
                    return _NEGONE, _NOUGHT
                else:
                    return None
            else:
                return None
            # if indice1[1] == indice2[1]:
            #     if key1 > key2:
            #         return _NEGONE, _NOUGHT
            #     else:
            #         return None
            # elif indice2[1] == SpinOneHalf.UP:
            #     return _NEGONE, _NOUGHT
            # else:
            #     return None
        elif char2 == _C_:
            if key1 > key2:
                if indice1[1] == indice2[1]:
                    return _NEGONE, delta
                else:
                    return _NEGONE, _NOUGHT
            else:
                return None
    elif char1 == _C_:
        if char2 == _P_DAG:
            if indice1[1] == SpinOneHalf.UP:
                return _UNITY, delta * spec.c_dag[indice1[0], SpinOneHalf.DOWN]
            elif indice1[1] == SpinOneHalf.DOWN:
                return _UNITY, \
                    _NEGONE * delta * spec.c_dag[indice1[0], SpinOneHalf.UP]
            else:
                assert False
        elif char2 == _N_UP_:
            if indice1[1] == SpinOneHalf.UP:
                return _UNITY, delta * spec.c_[indice1]
            elif indice1[1] == SpinOneHalf.DOWN:
                return _UNITY, _NOUGHT
            else:
                assert False
        elif char2 == _N_DN_:
            if indice1[1] == SpinOneHalf.DOWN:
                return _UNITY, delta * spec.c_[indice1]
            elif indice1[1] == SpinOneHalf.UP:
                return _UNITY, _NOUGHT
            else:
                assert False
        elif char2 == _N_:
            return _UNITY, delta * spec.c_[indice1]
        elif char2 == _P_:
            return _UNITY, _NOUGHT
        elif char2 == _S_P:
            if indice1[1] == SpinOneHalf.UP:
                return _UNITY, delta * spec.c_[indice1[0], SpinOneHalf.DOWN]
            elif indice1[1] == SpinOneHalf.DOWN:
                return _UNITY, _NOUGHT
            else:
                assert False
        elif char2 == _S_Z:
            return _UNITY, _HALF * delta * spec.c_[indice1]
        elif char2 == _S_M:
            if indice1[1] == SpinOneHalf.UP:
                return _UNITY, _NOUGHT
            elif indice1[1] == SpinOneHalf.DOWN:
                return _UNITY, delta * spec.c_[indice1[0], SpinOneHalf.UP]
            else:
                assert False
        elif char2 == _C_DAG:
            if key1 >= key2:
                if indice1[1] == indice2[1]:
                    return _NEGONE, delta
                else:
                    return _NEGONE, _NOUGHT
            else:
                return None
        elif char2 == _C_:
            if key1 > key2:
                return _NEGONE, _NOUGHT
            elif key1 == key2:
                if indice1[1] == SpinOneHalf.UP:
                    return _NEGONE, _NOUGHT
                else:
                    return None
            else:
                return None
            # if indice1[1] == indice2[1]:
            #     if key1 > key2:
            #         return _NEGONE, _NOUGHT
            #     else:
            #         return None
            # elif indice2[1] == SpinOneHalf.DOWN:
            #     return _NEGONE, _NOUGHT
            # else:
            #     return None
    else:
        # return None
        assert False


def _nonzero_by_cartan(term: Term):
    """If the term is zero because of the cartan in it.
    NOTE: the nonzero by cartan filter should be used only when the terms are
    already in a canonical/normal order so that the Pdag N, and P terms
    are kept together.
    """

    raise_ = AGPFermi.PAIRING_RAISE
    cartan = AGPFermi.PAIRING_CARTAN
    lower = AGPFermi.PAIRING_LOWER

    raise_indices = set()
    cartan_indices = set()

    for vec in term.vecs:
        base = vec.base
        indices = vec.indices

        if base == raise_:
            raise_indices.add(indices)
        elif base == cartan:
            if indices in raise_indices:
                return False
            cartan_indices.add(indices)
        elif base == lower:
            if indices in cartan_indices:
                return False

        continue

    return True


def _nonzero_by_nilp(term: Term):
    """Need a function to filter terms based on nilpotency of fermion operators
    """
    vecs = term.vecs
    cartans = (AGPFermi.PAIRING_CARTAN, AGPFermi.SPIN_CARTAN)
    return all(
        (
            not ((vecs[i].base not in cartans) and (vecs[i] == vecs[i+1]))
        )
        for i in range(0, len(vecs)-1)
    )


def _no_fermi_filter(term, spec: _AGPFSpec):
    """Filter to drop all terms with c_dag or c_ operators
    """

    vecs = term.vecs
    cr = spec.c_dag.base
    an = spec.c_.base

    if len(vecs) == 0:
        return True

    for v in vecs:
        if v.base in (cr, an):
            return False

    return True


def _even_fermi_filter(term, spec: _AGPFSpec):
    """Filter function to throw away the terms containing odd number
    of fermion operators
    """
    vecs = term.vecs

    cr = spec.c_dag.base
    an = spec.c_.base

    n_fermi = 0

    for v in vecs:
        if (v.base == cr) or (v.base == an):
            n_fermi += 1

    if n_fermi % 2 == 0:
        return True
    else:
        return False


def _get_su2_vecs(term: Term, spec: _AGPFSpec):
    """Given a term with a list of vectors, extract the obvious BCS vectors
    NOTE: This bind function assumes that the term has already been simplified
    NOTE2: We just ignore the SP and SM terms in extracting su2
    """
    vecs = term.vecs
    amp = term.amp
    int_vecs = []
    new_vecs = []

    Pdag = spec.Pdag
    P = spec.P
    N = spec.N

    N_up = spec.Nup
    N_dn = spec.Ndn

    SP = spec.S_p
    SM = spec.S_m
    SZ = spec.S_z

    cr = (spec.c_dag.base, spec.c_dag.indices[0])
    an = (spec.c_.base, spec.c_.indices[0])

    # Get rid of the trivial case where no SU2 vector is possible.
    if len(vecs) <= 1:
        new_vecs = vecs
        return [Term(sums=term.sums, amp=amp, vecs=new_vecs)]

    # Now onto the more involved situations:
    # First, let us extract all the number and pair-annihilation operators
    i = 0
    while i < (len(vecs)-1):

        v1 = (vecs[i].base, vecs[i].indices[0])
        v2 = (vecs[i+1].base, vecs[i+1].indices[0])

        if (v1 == cr):
            if (v2 == an):
                if (vecs[i].indices[1:] == vecs[i+1].indices[1:]):
                    if vecs[i].indices[2] == SpinOneHalf.UP:
                        int_vecs.append(N_up[vecs[i].indices[1]])
                    elif vecs[i].indices[2] == SpinOneHalf.DOWN:
                        int_vecs.append(N_dn[vecs[i].indices[1]])
                    i += 2
                    continue
                # elif (vecs[i].indices[1] == vecs[i+1].indices[1]):
                #     if vecs[i].indices[2] == SpinOneHalf.UP:
                #         int_vecs.append(SP[vecs[i].indices[1]])
                #         i += 2
                #         continue
                #     else:
                #         int_vecs.append(SM[vecs[i].indices[1]])
                #         i += 2
                #         continue
                else:
                    int_vecs.append(vecs[i])
                    i += 1
                    continue
            elif (v2 == cr):
                # Do not consider the pair-creation operators yet
                int_vecs.append(vecs[i])
                i += 1
                continue
            else:
                raise ValueError('Input term is not simplified')

        elif (v1 == an):
            if (v2 == an):
                if (vecs[i].indices[1] == vecs[i+1].indices[1]):
                    # if both annihilation, and have same lattice index,
                    #   then assuming the term is simplified, the spins
                    #    must be opposite (first one would be DOWN)
                    int_vecs.append(P[vecs[i].indices[1]])
                    i += 2
                    continue
                else:
                    int_vecs.append(vecs[i])
                    i += 1
                    continue
            else:
                int_vecs.append(vecs[i])
                i += 1
                continue
        else:
            int_vecs.append(vecs[i])
            i += 1
            continue

    # If the last two operators did not map, we are left with
    # the last fermion operator
    if i == len(vecs) - 1:
        int_vecs.append(vecs[i])

    # Now map all the pair-creation operators
    i = 0
    while i < (len(int_vecs)-1):

        v1 = (int_vecs[i].base, int_vecs[i].indices[0])
        v2 = (int_vecs[i+1].base, int_vecs[i+1].indices[0])

        if (v1 == cr):
            if (v2 == cr):
                if (int_vecs[i].indices[1] == int_vecs[i+1].indices[1]):
                    # if both creation, and have same lattice index,
                    #   then assuming term is simplified, the spins must be
                    #   opposite.
                    new_vecs.append(Pdag[int_vecs[i].indices[1]])
                    i += 2
                    continue
                else:
                    new_vecs.append(int_vecs[i])
                    i += 1
                    continue
            else:
                new_vecs.append(int_vecs[i])
                i += 1
                continue
                # raise ValueError('Input term is not simplified')
        else:
            new_vecs.append(int_vecs[i])
            i += 1
            continue

    # If the last two operators did not map, we are left with
    # the last fermion operator
    if i == len(int_vecs) - 1:
        new_vecs.append(int_vecs[i])

    return [Term(sums=term.sums, amp=amp, vecs=new_vecs)]


def _get_fermi_partitions(term: Term, spec: _AGPFSpec):
    """Given a term with a list of fermionic vectors, extract the various
    partitions of N, Pdag and P kind of terms.

    Assumes: All possible simplification and Pairing / SU2 extraction
    has been performed, i.e. all the indices are distinct.
    """

    vecs = term.vecs
    amp = term.amp
    fermi_indcs = []

    cr = spec.c_dag.base
    an = spec.c_.base

    # First extract all the indices of fermion operators
    for v in vecs:
        if v.base in (cr, an):
            fermi_indcs.append(v.indices[1])

    if len(fermi_indcs) == 0:
        # if there are no fermion operators, return the term as it is
        return [Term(sums=term.sums, amp=amp, vecs=vecs)]
    else:
        # if there are fermion operators, then get all the possible
        # partitions of indices
        deltas_partns = list(_generate_partitions(fermi_indcs))

    # get a list of even partitions, i.e. throw away partitions involving
    # odd number of indices
    evens = [
        all(
            len(deltas_partns[i][j]) % 2 == 0
            for j in range(len(deltas_partns[i]))
        ) for i in range(len(deltas_partns))
    ]

    # Now iterate through the list of partitions and form the deltas expression
    # for the amplitude
    delta_amp = Integer(0)

    for i in range(len(deltas_partns)):
        # if statement for even partitions
        if evens[i]:
            pt = deltas_partns[i]
        else:
            continue

        # Form the delta expressions
        delta_intmd = Integer(1)
        for i in range(len(pt)):
            delta_intmd *= _construct_deltas(pt[i])
            if i > 0:
                delta_intmd *= (
                    1 - KroneckerDelta(pt[i][0], pt[i-1][0])
                )
        delta_amp += delta_intmd

    new_amp = amp * delta_amp

    return [Term(sums=term.sums, amp=new_amp, vecs=vecs)]


def _construct_deltas(indices):
    """Given a list of indices, form all possible KroneckerDelta pairs
    """
    del_pairs = Integer(1)
    for i in range(len(indices)-1):
        del_pairs *= KroneckerDelta(indices[i], indices[i+1])

    return del_pairs


def _generate_partitions(indices):
    """A general function that generates all possible partitions for
    a given set of indices
    """

    if len(indices) == 1:
        yield [indices]
        return

    first = indices[0]
    for smaller in _generate_partitions(indices[1:]):
        # insert `first' in each of sub-partitions subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        # put `first' in its own subset
        yield [[first]] + smaller


def _delta_map(input_substs):
    """Function takes in a dictionary of substitutions and creates a set of
    connected lists of indices of the KroneckerDeltas
    """

    substs = input_substs.copy()

    # First get the keys of the dictionary
    dict_keys = list(substs.keys())

    # Start with the first key - see where it points to and see if the
    # substituting index is itself in the keys

    delta_lists = []
    new_list = []

    # Starting Key
    key = dict_keys[-1]
    new_list = [key, ]

    while dict_keys:

        # First form a small list with the first two indices
        val = substs[key]
        new_list.append(val)

        # remove these entries from the dict
        dict_keys.remove(key)
        del substs[key]

        if val in dict_keys:

            # Update the elements
            key = val
            continue

        else:

            # Can't extend anymore
            new_set = set(new_list)
            lst_added = False

            if not delta_lists:
                delta_lists.append(new_set)
            else:
                lst_ind = 0
                for lst in delta_lists:
                    ovlp = list(set(lst) & set(new_set))
                    if ovlp:
                        delta_lists[lst_ind] = delta_lists[lst_ind].union(
                            new_set
                        )
                        lst_added = True
                        break
                    lst_ind += 1
                if lst_added is False:
                    delta_lists.append(new_set)

            if dict_keys:
                key = dict_keys[-1]
                new_list = [key, ]

    return delta_lists


def _canonicalize_indices(term: Term, spec: _AGPFSpec):
    """Here, we canonicalize the free indices in the tensor expressions - that
    is replace the higher key indices with lower key ones everywhere
    """

    # get the new term and substs
    new_amp, substs = _try_simpl_unresolved_deltas(term.amp)

    # check for overlap in the dlists and unique_indices
    # if any of the maps contain two indices that are supposed to be unique,
    # it should return zero/empty term.
    unique_list = spec.unique_ind

    # New substitutions based on the chain of delta
    new_substs = {}

    if substs:
        dlists = _delta_map(substs)

        for s1 in dlists:
            s1_sorted = list(ordered(s1))
            j = 1
            while j < len(s1_sorted):
                new_substs[s1_sorted[j]] = s1_sorted[0]
                j += 1

            if not unique_list:
                continue

            for s2 in unique_list:
                if len(list(s1 & s2)) > 1:
                    return []
                else:
                    continue

    # construct the new term
    new_term = term.subst(new_substs)

    return [Term(sums=new_term.sums, amp=new_amp, vecs=new_term.vecs)]


def _try_simpl_unresolved_deltas(amp: Expr):
    """Try some simplification on unresolved deltas.

    This function aims to normalize the usage of free indices in the amplitude
    when a delta factor is present to require their equality.

    TODO: Unify the treatment here and the treatment for summation dummies.
    """

    substs = {}
    if not (isinstance(amp, Mul) or isinstance(amp, KroneckerDelta)):
        return amp, substs

    deltas = _UNITY
    others = _UNITY

    if isinstance(amp, KroneckerDelta):
        arg1, arg2 = amp.args

        # Here, only the simplest case is treated, a * x = b * y, with a, b
        # being numbers and x, y being atomic symbols.  One of the symbols
        # can be missing.  But not both, since SymPy will automatically
        # resolve a delta between two numbers.

        factor1, symb1 = _parse_factor_symb(arg1)
        factor2, symb2 = _parse_factor_symb(arg2)

        if factor1 is not None and factor2 is not None:
            if symb1 is None:
                assert symb2 is not None
                arg1 = symb2
                arg2 = factor1 / factor2
            elif symb2 is None:
                assert symb1 is not None
                arg1 = symb1
                arg2 = factor2 / factor1
            elif sympy_key(symb1) < sympy_key(symb2):
                arg1 = symb2
                arg2 = factor1 * symb1 / factor2
            else:
                arg1 = symb1
                arg2 = factor2 * symb2 / factor1
            substs[arg1] = arg2

        deltas *= KroneckerDelta(arg1, arg2)

    else:
        for i in amp.args:
            if isinstance(i, KroneckerDelta):
                arg1, arg2 = i.args

                # Here, only the simplest case is treated, a * x = b * y,
                # with a, b being numbers and x, y being atomic symbols.
                # One of the symbols can be missing.  But not both,
                # since SymPy will automatically resolve a delta between
                # two numbers.

                factor1, symb1 = _parse_factor_symb(arg1)
                factor2, symb2 = _parse_factor_symb(arg2)
                if factor1 is not None and factor2 is not None:
                    if symb1 is None:
                        assert symb2 is not None
                        arg1 = symb2
                        arg2 = factor1 / factor2
                    elif symb2 is None:
                        assert symb1 is not None
                        arg1 = symb1
                        arg2 = factor2 / factor1
                    elif sympy_key(symb1) < sympy_key(symb2):
                        arg1 = symb2
                        arg2 = factor1 * symb1 / factor2
                    else:
                        arg1 = symb1
                        arg2 = factor2 * symb2 / factor1

                    # Now update the substs dictionary
                    if arg1 not in list(substs.keys()):
                        substs[arg1] = arg2
                    else:
                        # if the arg1 exists in the key,
                        # multiply with the new delta
                        deltas *= KroneckerDelta(arg2, substs[arg1])
                deltas *= KroneckerDelta(arg1, arg2)
            else:
                others *= i

    others = others.xreplace(substs)
    return deltas * others, substs


def _parse_factor_symb(expr: Expr):
    """Parse a number times a symbol.

    When the expression is of that form, that number and the only symbol is
    returned.  For plain numbers, the symbol part is none.  For completely
    non-compliant expressions, a pair of none is going to be returned.
    """

    if isinstance(expr, Symbol):
        return _UNITY, expr
    elif isinstance(expr, Number):
        return expr, None
    elif isinstance(expr, Mul):
        factor = _UNITY
        symb = None
        for i in expr.args:
            if isinstance(i, Number):
                factor *= i
            elif isinstance(i, Symbol):
                if symb is None:
                    symb = i
                else:
                    return None, None
            else:
                return None, None
        return factor, symb
    else:
        return None, None


_UNITY = Integer(1)
_HALF = Rational(1, 2)
_NOUGHT = Integer(0)
_NEGONE = Integer(-1)
_NEGHALF = -Rational(1, 2)
_TWO = Integer(2)
