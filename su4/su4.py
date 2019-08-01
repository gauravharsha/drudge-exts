"""Drudge for SU(4) Lie algebra."""

import collections
import functools
import operator

from sympy import Integer, KroneckerDelta, Rational, Symbol

from drudge import *
from drudge.genquad import GenQuadDrudge
from drudge.term import Vec
from drudge.utils import sympy_key


class SU4LatticeDrudge(GenQuadDrudge):
    """Drudge for a lattice of SU(4) algebras, constructed strictly to cater
    the needs of Coupled Cluster with Thermofield Dynamics (TFD) in Lipkin.
    
    DESCRIPTION TO BE VERIFIED
    This drudge has the commutation rules for SU(4) algebras in Cartan-Weyl form
    (Ladder operators).  Here both the shift and Cartan operators can have
    additional *lattice indices*.  Operators on different lattice sites always
    commute.

    The the normal-ordering operation would try to put raising operators before
    the Cartan operators, which come before the lowering operators.

    """

    DEFAULT_CARTAN1 = Vec('J^z')
    DEFAULT_CARTAN2 = Vec('K^z')
    DEFAULT_RAISE1 = Vec('J^+')
    DEFAULT_RAISE2 = Vec('K^+')
    DEFAULT_LOWER1 = Vec('J^-')
    DEFAULT_LOWER2 = Vec('K^-')
    DEFAULT_Ypp = Vec('Y^{++}')
    DEFAULT_Ypm = Vec('Y^{+-}')
    DEFAULT_Ymp = Vec('Y^{-+}')
    DEFAULT_Ymm = Vec('Y^{--}')
    DEFAULT_Yzz = Vec('Y^{zz}')
    DEFAULT_Ypz = Vec('Y^{+z}')
    DEFAULT_Yzp = Vec('Y^{zp}')
    DEFAULT_Ymz = Vec('Y^{-z}')
    DEFAULT_Yzm = Vec('Y^{z-}')
    
    def __init__(
            self, ctx, root=Integer(1),
            cartan1=DEFAULT_CARTAN1, raise1=DEFAULT_RAISE1, lower1=DEFAULT_LOWER1,
            cartan2=DEFAULT_CARTAN2, raise2=DEFAULT_RAISE2, lower2=DEFAULT_LOWER2,
            ypp=DEFAULT_Ypp, ypm=DEFAULT_Ypm, ymp=DEFAULT_Ymp, ymm=DEFAULT_Ymm,
            yzz=DEFAULT_Yzz, ypz=DEFAULT_Ypz, yzp=DEFAULT_Yzp, ymz=DEFAULT_Ymz, yzm=DEFAULT_Yzm,
            **kwargs
    ):
        """Initialize the drudge.

        Parameters
        ----------

        ctx
            The Spark context for the drudge.

        2 cartan
            The basis operator for the Cartan subalgebra (:math:`J^z` operator
            for spin problem).  It is registered in the name archive by the
            first letter in its label followed by an underscore.

        2 raise
            The raising operator.  It is also also registered in the name
            archive by the first letter in its label followed by ``_p``.

        2lower
            The lowering operator, registered by the first letter followed by
            ``_m``.
        
        9 Cross operators (Y_ab)
            Mix of the two SU(2)'s described by the cartans, raise and lower
            operators above.

        root
            The coefficient for the commutator between the Cartan and shift
            operators.

        norm
            (Feature removed from SU2 -> SU4, seems unused)
            The coefficient for the commutator between the raising and lowering
            operators.

        kwargs
            All other keyword arguments are given to the base class
            :py:class:`GenQuadDrudge`.

        """
        super().__init__(ctx, **kwargs)
        
        self.cartan1 = cartan1
        self.cartan2 = cartan2
        self.raise1 = raise1
        self.lower1 = lower1
        self.raise2 = raise2
        self.lower2 = lower2
        self.ypp = ypp
        self.ymm = ymm
        self.yzz = yzz
        self.ypm = ypm
        self.ymp = ymp
        self.ypz = ypz
        self.yzp = yzp
        self.ymz = ymp
        self.yzm = yzm
        
        self.set_name(**{
            cartan1.label[0] + '_': cartan1,
            cartan2.label[0] + '_': cartan2,
            raise1.label[0] + '_p': raise1,
            raise2.label[0] + '_p': raise2,
            lower1.label[0] + '_m': lower1,
            lower2.label[0] + '_m': lower2,
            ypp.label[0] + '_pp': ypp,
            ymm.label[0] + '_mm': ymm,
            yzz.label[0] + '_zz': yzz,
            ypm.label[0] + '_pm': ypm,
            ymp.label[0] + '_mp': ymp,
            ypz.label[0] + '_pz': ypz,
            yzp.label[0] + '_zp': yzp,
            ymz.label[0] + '_mz': ymz,
            yzm.label[0] + '_zm': yzm
        })
        
        spec = _SU4Spec(
            cartan1=cartan1, raise1=raise1, lower1=lower1, root=root, 
            cartan2=cartan2, raise2=raise2, lower2=lower2, 
            ypp=ypp, ymm=ymm, yzz=yzz, ypm=ypm,
            ymp=ymp, ypz=ypz, yzp=yzp, ymz=ymz, yzm=yzm
        )
        self._spec = spec
        
        self._swapper = functools.partial(_swap_su4, spec=spec)
    
    @property
    def swapper(self) -> GenQuadDrudge.Swapper:
        """The swapper for the spin algebra."""
        return self._swapper
    
    def eval_exp(self,h_tsr: Tensor, n1: Symbol):
        """Function to evaluate the expectation on hartree fock ground state of
        n-particle Lipkin Hamiltonian
        """
        return h_tsr.bind(lambda x: get_vev_of_term(x,n1) )



_SU4Spec = collections.namedtuple('_SU4Spec',[
    'cartan1',
    'raise1',
    'lower1',
    'root',
    'cartan2',
    'raise2',
    'lower2',
    'ypp',
    'ymm',
    'yzz',
    'ypm',
    'ymp',
    'ypz',
    'yzp',
    'ymz',
    'yzm'
])

def _swap_su4(vec1: Vec, vec2: Vec, depth=None, *,spec: _SU4Spec):
    """Swap two vectors based on the TFD SU4 rules
    Here, we introduce an additional input parameter 'depth' which is never
    specified by the user. Rather, it is put to make use os the anti-symmetric 
    nature of the commutation relations and make the function def compact. 
    """
    if depth is None:
        depth = 1
    
    char1, indice1, key1 = _parse_vec(vec1,spec)
    char2, indice2, key2 = _parse_vec(vec2,spec)
    
    if len(indice1) != len(indice2):
        raise ValueError(
            'Invalild SU4 generators on lattice', (vec1, vec2),
            'Incompatible number of lattice indices'
        )
    
    delta = functools.reduce(operator.mul, (
        KroneckerDelta(i,j) for i,j in zip(indice1,indice2)
    ), _UNITY)
    
    root = spec.root
    
    if char1 == _YPP:
        
        if char2 == _YPP:
            if key1 < key2:
                return None
            elif key1 > key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _RAISE1:
        
        if char2 == _YPP:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE1:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _RAISE2:
        
        if char2 == _YPP:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE1:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE2:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else: 
            return None
    
    elif char1 == _YPZ:
        
        if char2 == _YPP:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE1:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE2:
            return _UNITY, root * delta * spec.ypp[indice1]
        elif char2 == _YPZ:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _YZP:
        
        if char2 == _YPP:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE1:
            return _UNITY, root * delta * spec.ypp[indice1]
        elif char2 == _RAISE2:
            return _UNITY, _NOUGHT
        elif char2 == _YPZ:
            return _UNITY, _NOUGHT
        elif char2 == _YZP:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _YPM:
        
        if char2 == _YPP:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE1:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE2:
            return _UNITY, _NEGTWO * delta * spec.ypz[indice1]
        elif char2 == _YPZ:
            return _UNITY, _NOUGHT
        elif char2 == _YZP:
            return _UNITY, _NEGHALF * delta * spec.raise1[indice1]
        elif char2 == _YPM:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _YZZ: 
        
        if char2 == _YPP:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE1:
            return _UNITY, root * delta * spec.ypz[indice1]
        elif char2 == _RAISE2:
            return _UNITY, root * delta * spec.yzp[indice1]
        elif char2 == _YPZ:
            return _UNITY, _QUARTER * delta * spec.raise1[indice1]
        elif char2 == _YZP:
            return _UNITY, _QUARTER * delta * spec.raise2[indice1]
        elif char2 == _YPM:
            return _UNITY, _NOUGHT
        elif char2 == _YZZ:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _CARTAN1:
        
        if char2 == _YPP:
            return _UNITY, root * delta * spec.ypp[indice1]
        elif char2 == _RAISE1:
            return _UNITY, root * delta * spec.raise1[indice1]
        elif char2 == _RAISE2:
            return _UNITY, _NOUGHT
        elif char2 == _YPZ:
            return _UNITY, root * delta * spec.ypz[indice1]
        elif char2 == _YZP:
            return _UNITY, _NOUGHT
        elif char2 == _YPM:
            return _UNITY, root * delta * spec.ypm[indice1]
        elif char2 == _YZZ:
            return _UNITY, _NOUGHT
        elif char2 == _CARTAN1:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _CARTAN2:
        
        if char2 == _YPP:
            return _UNITY, root * delta * spec.ypp[indice1]
        elif char2 == _RAISE1:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE2:
            return _UNITY, root * delta * spec.raise2[indice1]
        elif char2 == _YPZ:
            return _UNITY, _NOUGHT
        elif char2 == _YZP:
            return _UNITY, root * delta * spec.yzp[indice1]
        elif char2 == _YPM:
            return _UNITY, -root * delta * spec.ypm[indice1]
        elif char2 == _YZZ:
            return _UNITY, _NOUGHT
        elif char2 == _CARTAN1:
            return _UNITY, _NOUGHT
        elif char2 == _CARTAN2:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _YMP:
        
        if char2 == _YPP:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE1:
            return _UNITY, _NEGTWO * delta * spec.yzp[indice1]
        elif char2 == _RAISE2:
            return _UNITY, _NOUGHT
        elif char2 == _YPZ:
            return _UNITY, _NEGHALF * delta * spec.raise2[indice1]
        elif char2 == _YZP:
            return _UNITY, _NOUGHT
        elif char2 == _YPM:
            return _UNITY, delta * (spec.cartan2[indice1] - spec.cartan1[indice1])
        elif char2 == _YZZ:
            return _UNITY, _NOUGHT
        elif char2 == _CARTAN1:
            return _UNITY, root * delta * spec.ymp[indice1]
        elif char2 == _CARTAN2:
            return _UNITY, -root * delta * spec.ymp[indice1]
        elif char2 == _YMP:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _YMZ:
        
        if char2 == _YPP:
            return _UNITY, Rational(1,2) * delta * spec.raise2[indice1]
        elif char2 == _RAISE1:
            return _UNITY, _NEGTWO * delta * spec.yzz[indice1]
        elif char2 == _RAISE2:
            return _UNITY, root * delta * spec.ymp[indice1]
        elif char2 == _YPZ:
            return _UNITY, _NEGHALF * delta * spec.cartan1[indice1]
        elif char2 == _YZP:
            return _UNITY, _NOUGHT
        elif char2 == _YPM:
            return _UNITY, _NEGHALF * delta * spec.lower2[indice1]
        elif char2 == _YZZ:
            return _UNITY, _QUARTER * delta * spec.lower1[indice1]
        elif char2 == _CARTAN1:
            return _UNITY, root * delta * spec.ymz[indice1]
        elif char2 == _CARTAN2:
            return _UNITY, _NOUGHT
        elif char2 == _YMP:
            return _UNITY, _NOUGHT
        elif char2 == _YMZ:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _YZM:
        
        if char2 == _YPP:
            return _UNITY, Rational(1,2) * delta * spec.raise1[indice1]
        elif char2 == _RAISE1:
            return _UNITY, root * delta * spec.ypm[indice1]
        elif char2 == _RAISE2:
            return _UNITY, _NEGTWO * delta * spec.yzz[indice1]
        elif char2 == _YPZ:
            return _UNITY, _NOUGHT
        elif char2 == _YZP:
            return _UNITY, _NEGHALF * delta * spec.cartan2[indice1]
        elif char2 == _YPM:
            return _UNITY, _NOUGHT
        elif char2 == _YZZ:
            return _UNITY, _QUARTER * delta * spec.lower2[indice1]
        elif char2 == _CARTAN1:
            return _UNITY, _NOUGHT
        elif char2 == _CARTAN2:
            return _UNITY, root * delta * spec.yzm[indice1]
        elif char2 == _YMP:
            return _UNITY, _NEGHALF * delta * spec.lower1[indice1]
        elif char2 == _YMZ:
            return _UNITY, _NOUGHT
        elif char2 == _YZM:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _LOWER1:
        
        if char2 == _YPP:
            return _UNITY, _NEGTWO * delta * spec.yzp[indice1]
        elif char2 == _RAISE1:
            return _UNITY, _NEGTWO * delta * spec.cartan1[indice1]
        elif char2 == _RAISE2:
            return _UNITY, _NOUGHT
        elif char2 == _YPZ:
            return _UNITY, _NEGTWO * delta * spec.yzz[indice1]
        elif char2 == _YZP:
            return _UNITY, root * delta * spec.ymp[indice1]
        elif char2 == _YPM:
            return _UNITY, _NEGTWO * delta * spec.yzm[indice1]
        elif char2 == _YZZ:
            return _UNITY, root * delta * spec.ymz[indice1]
        elif char2 == _CARTAN1:
            return _UNITY, root * delta * spec.lower1[indice1]
        elif char2 == _CARTAN2:
            return _UNITY, _NOUGHT
        elif char2 == _YMP:
            return _UNITY, _NOUGHT
        elif char2 == _YMZ:
            return _UNITY, _NOUGHT
        elif char2 == _YZM:
            return _UNITY, root * delta * spec.ymm[indice1]
        elif char2 == _LOWER1:
            if key1 < key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _LOWER2:
        
        if char2 == _YPP:
            return _UNITY, _NEGTWO * delta * spec.ypz[indice1]
        elif char2 == _RAISE1:
            return _UNITY, _NOUGHT
        elif char2 == _RAISE2:
            return _UNITY, _NEGTWO *delta * spec.cartan2[indice1]
        elif char2 == _YPZ:
            return _UNITY, root * delta * spec.ypm[indice1]
        elif char2 == _YZP:
            return _UNITY, _NEGTWO * delta * spec.yzz[indice1]
        elif char2 == _YPM:
            return _UNITY, _NOUGHT
        elif char2 == _YZZ:
            return _UNITY, root * delta * spec.yzm[indice1]
        elif char2 == _CARTAN1:
            return _UNITY, _NOUGHT
        elif char2 == _CARTAN2:
            return _UNITY, root * delta * spec.lower2[indice1]
        elif char2 == _YMP:
            return _UNITY, _NEGTWO * delta * spec.ymz[indice1]
        elif char2 == _YMZ:
            return _UNITY, root * delta * spec.ymm[indice1]
        elif char2 == _YZM:
            return _UNITY, _NOUGHT
        elif char2 == _LOWER1:
            return _UNITY, _NOUGHT
        elif char2 == _LOWER2:
            if key1<key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    
    elif char1 == _YMM:
        
        if char2 == _YPP:
            return _UNITY, -root * delta * (spec.cartan1[indice1] + spec.cartan2[indice1])
        elif char2 == _RAISE1:
            return _UNITY, _NEGTWO * delta * spec.yzm[indice1]
        elif char2 == _RAISE2:
            return _UNITY, _NEGTWO * delta * spec.ymz[indice1]
        elif char2 == _YPZ:
            return _UNITY, Rational(1,2) * delta * spec.lower2[indice1]
        elif char2 == _YZP:
            return _UNITY, Rational(1,2) * delta * spec.lower1[indice1]
        elif char2 == _YPM:
            return _UNITY, _NOUGHT
        elif char2 == _YZZ:
            return _UNITY, _NOUGHT
        elif char2 == _CARTAN1:
            return _UNITY, root * delta * spec.ymm[indice1]
        elif char2 == _CARTAN2:
            return _UNITY, root * delta * spec.ymm[indice1]
        elif char2 == _YMP:
            return _UNITY, _NOUGHT
        elif char2 == _YMZ:
            return _UNITY, _NOUGHT
        elif char2 == _YZM:
            return _UNITY, _NOUGHT
        elif char2 == _LOWER1:
            return _UNITY, _NOUGHT
        elif char2 == _LOWER2:
            return _UNITY, _NOUGHT
        elif char2 == _YMM:
            if key1<key2:
                return _UNITY, _NOUGHT
            else:
                return None
        else:
            return None
    else:
        assert False



_RAISE1 = 0
_CARTAN1 = 1
_LOWER1 = 2
_RAISE2 = 3
_CARTAN2 = 4
_LOWER2 = 5
_YPP = 6
_YMM = 7
_YZZ = 8
_YPM = 9
_YMP = 10
_YPZ = 11
_YZP = 12
_YMZ = 13
_YZM = 14


def _parse_vec(vec, spec: _SU4Spec):
    """Get the character, lattice indices, and indices keys of the vector.
    """
    base = vec.base
    if base == spec.cartan1:
        char = _CARTAN1
    elif base == spec.raise1:
        char = _RAISE1
    elif base == spec.lower1:
        char = _LOWER1
    elif base == spec.cartan2:
        char = _CARTAN2
    elif base == spec.raise2:
        char = _RAISE2
    elif base == spec.lower2:
        char = _LOWER2
    elif base == spec.ypp:
        char = _YPP
    elif base == spec.ymm:
        char = _YMM
    elif base == spec.yzz:
        char = _YZZ
    elif base == spec.ypm:
        char = _YPM
    elif base == spec.ymp:
        char = _YMP
    elif base == spec.ypz:
        char = _YPZ
    elif base == spec.yzp:
        char = _YZP
    elif base == spec.ymz:
        char = _YMZ
    elif base == spec.yzm:
        char = _YZM
    else:
        raise ValueError('Unexpected vector for SU2 algebra', vec)
    
    indices = vec.indices
    keys = tuple(sympy_key(i) for i in indices)
    
    return char, indices, keys

_CARTAN1 = SU4LatticeDrudge.DEFAULT_CARTAN1
_CARTAN2 = SU4LatticeDrudge.DEFAULT_CARTAN2
_YZZ = SU4LatticeDrudge.DEFAULT_Yzz

def get_vev_of_term(term,n1):
    """Return the vev of a given term"""
    vecs = term.vecs
    t_amp = term.amp #coefficient of the term itself

    # take care of the scalar term
    if len(vecs)==0:
        return term

    for i in vecs:
        if i.base == _CARTAN1:
            t_amp = t_amp*(-n1/2)
        elif i.base == _CARTAN2:
            t_amp = t_amp*(-n1/2)
        elif i.base == _YZZ:
            t_amp = t_amp*(n1/4)
        else:
            return []
            break
    return [Term(sums=term.sums,amp = t_amp,vecs=())]

_QUARTER = Rational(1,4)
_NEGHALF = -Rational(1,2)
_NEGTWO = -Integer(2)
_UNITY = Integer(1)
_NOUGHT = Integer(0)
