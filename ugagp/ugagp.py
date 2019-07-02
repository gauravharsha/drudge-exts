"""
Drudge for reduced AGP - PBCS Hamiltonian.
"""
import collections, functools, operator, re
import pdb

from sympy import Integer, Symbol, IndexedBase, KroneckerDelta, factorial, Function, srepr
from sympy.utilities.iterables import default_sort_key

from drudge import Tensor
from drudge import PartHoleDrudge, SpinOneHalfPartHoleDrudge
from drudge.canon import IDENT,NEG
from drudge.canonpy import Perm
from drudge.term import Vec, Range, Term
from drudge.utils import sympy_key
from drudge.genquad import GenQuadDrudge


class ProjectedBCSDrudge(GenQuadDrudge):
    r"""Drudge to manipulate and work with new kind of creation, cartan and
    annihilation operators defined with respect to the Projected BCS reference,
    alternatively known as the AGP state. The three generators are:
    :math:`\mathbf{D}_{p,q}^\dagger`, :math:`\mathbf{N}_p` and 
    :math:`\mathbf{D}_{pq}` which are defined as a local rotation of the
    unitary group generators.

    While the Projected BCS drudge is self contained, one would generally want to
    use it to study the pairing hamiltonian in the basis of these AGP operators.
    And it is much faster to work in the basis of unitary group generators using the
    UnitaryGroupDrudge. Eventually one can map the final expressions into this
    Projected BCS Drudge and compute VEV with respect to the AGP states.
    
    The commutation rules are very complicated.

    Some definitions for the symbols are described here

    :math: `\sigma_{pq} = \frac{1}{\eta_p^2 - \eta_q^2}`

    :math: `\eta_p` defines the transformation

    :math: `\mathbf{D}_{pq} = \frac{\eta_p E^p_q - \eta_q E^q_p}{\eta_p + \eta_q}`

    """

    DEFAULT_CARTAN = Vec('N')
    DEFAULT_RAISE = Vec(r'D^\dagger')
    DEFAULT_LOWER = Vec('D')

    def __init__(
            self, ctx,
            all_orb_range=Range('A' ,0, Symbol('na')),
            all_orb_dumms=PartHoleDrudge.DEFAULT_ORB_DUMMS,
            cartan=DEFAULT_CARTAN, raise_=DEFAULT_RAISE, lower=DEFAULT_LOWER,
            **kwargs
    ):
        """Initialize the drudge object."""

        # Initialize the base su2 problem.
        super().__init__(ctx, **kwargs)

        # Set the range and dummies.
        self.all_orb_range = all_orb_range
        self.all_orb_dumms = tuple(all_orb_dumms)
        self.set_dumms(all_orb_range, all_orb_dumms)
        self.set_name(*self.all_orb_dumms)
        self.add_resolver({
            i: (all_orb_range) for i in all_orb_dumms
        })

        # Set the operator attributes
        self.cartan = cartan
        self.raise_ = raise_
        self.lower = lower

        self.set_symm(self.raise_,
            Perm([1,0],NEG),
            valence=2
        )
        self.set_symm(self.lower,
            Perm([1,0],NEG),
            valence=2
        )

        # Set the indexed objects attributes
        # NOTE: sigma is not assigned any symmetry because
        #       while sigma[p,q] = -sigma[q,p] for p != q,
        #       sigma[p,p] = not defined, and in fact, sigma[p,p] = sigma[p,p]
        # ---------------
        self.eta = _eta
        self.sigma = _sigma

        # Make additional name definition for the operators.
        self.set_name(**{
            cartan.label[0]+'_':cartan,
            raise_.label[0]+'_p':raise_,
            raise_.label[0]+'_dag':raise_,
            lower.label[0]+'_m':lower,
            lower.label[0]+'_':lower,
            'eta':self.eta,
            'sigma':self.sigma
        })

        self.set_name(cartan, lower, Ddag=raise_)
        
        # Defining spec for passing to an external function - the Swapper
        spec = _AGPSpec(
                cartan=cartan,raise_=raise_,lower=lower,
                eta=self.eta,sigma=self.sigma
        )
        self._spec = spec

        # set the Swapper
        self._swapper = functools.partial(_swap_agp, spec=spec)

        """
        # Definitions for translating to UGA
        self._uga_dr = UnitaryGroupDrudge(
            ctx, all_orb_range=self.all_orb_range, all_orb_dumms=self.all_orb_dumms
        )

        # Mapping from D to E
        _uga = self._uga_dr.E_
        gen_idx1, gen_idx2 = self.all_orb_dumms[:2]

        dm_pq_def = self.define(
            D_m, gen_idx1, gen_idx2,
            (self.eta[gen_idx1]*_uga[gen_idx1,gen_idx2] - self.eta[gen_idx2]*_uga[gen_idx2,gen_idx1])
        )

        dp_pq_def = self.define(
            D_p, gen_idx1, gen_idx2,
            (self.eta[gen_idx1]*_uga[gen_idx2,gen_idx1] - self.eta[gen_idx2]*_uga[gen_idx1,gen_idx2])
        )

        np_def = self.define(
            N_, gen_idx1, _uga[gen_idx1,gen_idx1]
        )

        self._agp2uga_defs = [
            dm_pq_def,
            dp_pq_def,
            np_def
        ]
        """

    @property
    def swapper(self) -> GenQuadDrudge.Swapper:
        """Swapper for the new AGP Algebra."""
        return self._swapper
    
    def _isAGP(self,term):
        """Returns True if the vectors in the term are generators of the AGP algebra, otherwise False"""
        vecs = term.vecs
        for v in vecs:
            if v.base not in (self.cartan, self.raise_, self.lower):
                raise ValueError('Unexpected generator of the AGP Algebra',v)
        return [Term(sums=term.sums, amp=term.amp, vecs=term.vecs)]

    def get_vev(self,h_tsr: Tensor):
        """Function to evaluate the expectation value of a normal
        ordered tensor 'h_tsr' with respect to the projected BCS
        ground state
            h_tsr = tensor whose VEV is to be evaluated
            Ntot = total number of orbitals available
        Note that here we follow the notation that Np = Number of Fermions and not number of pairs
        """
        ctan = self.cartan
        gam = IndexedBase(r'\gamma')

        def vev_of_term(term):
            """Return the VEV of a given term"""
            vecs = term.vecs
            t_amp = term.amp
            ind_list = []

            if len(vecs) == 0:
                return term
            elif all( v.base==ctan for v in vecs ):
                for i in vecs:
                    if set(i.indices).issubset(set(ind_list)):
                        t_amp = t_amp*2
                        # NOTE: This is only true when evaluating expectation over AGP,
                        # or any other seniority zero state
                    else:
                        ind_list.extend(list(i.indices))
            else:
                return []

            t_amp = t_amp*gam[ind_list]
            return [Term(sums=term.sums, amp = t_amp, vecs=())]

        return h_tsr.bind(vev_of_term)

    def asym_filter(self, terms, **kwargs):
        """Take the operators into normal order.

        Here, in addition to the common normal-ordering operation, we remove any
        term with a cartan operator followed by an lowering operator with the
        same index, and any term with a raising operator followed by a cartan
        operator with the same index.

        """

        def _dppfilter(term):
            """
            Function that filters out the D, Ddag terms with same index
            """
            vecs = term.vecs
            for v in vecs:
                indcs = vecs.indices
                if indcs[0]==indcs[1]:
                    return False
            return True

        noed = self.simplify(terms, **kwargs)
        noed = noed.filter(_dppfilter)
        noed = self.simplify(noed, **kwargs)
        return noed
        
    # NOTE: Initially, we assumed that only terms with same number of D and Ddag would contribute
    # But later, we realized that even D.D.Ddag can give some N term when normal ordered
    # This function and the comments are eventually meant to be deleted.

    # def _transl2uga(self, tensor: Tensor):
    #     """Translate a tensor object in terms of the fermion operators.

    #     This is an internal utility.  The resulting tensor has the internal
    #     '_uga_dr' drudge object as its owner.

    #     """
    #     return Tensor(
    #         self,
    #         tensor.subst_all(self._agp2uga_defs).terms
    #     )

class UnitaryGroupDrudge(GenQuadDrudge):
    r"""Drudge to manipulate and work with Unitary Group Generators defined as:
    ..math::

        E_{p,q} = c_{p,\uparrow}^\dagger c_{q,\uparrow} + \mathrm{h.c.}

    where the symbols :math: `p` and :math: `q` denote a general orbital index.
    This Unitary Group drudge comtains utilites to work with either the pairing
    Hamiltonian or any other system in the language of the unitary group gens.

    Without the presence of a reference, such as the Fermi sea or the pairing-
    ground state, there is no one way to define a canonical ordering for a string
    of :math: `E_{p,q}'\mathrm{s}`.

    Here we follow an lexicological or alphabetical ordering based on the first
    index followed by the second. For example, following is a representative of 
    the correct canonical ordering:
    ..math::
        E_{p,q} E_{p,r} E_{q,r} E_{r,p}

    """

    DEFAULT_GEN = Vec('E')

    def __init__(
            self, ctx,
            all_orb_range=Range('A' ,0, Symbol('na')),
            all_orb_dumms=PartHoleDrudge.DEFAULT_ORB_DUMMS, 
            energies=IndexedBase(r'\epsilon'), interact=IndexedBase('G'),
            uga=DEFAULT_GEN,
            **kwargs
    ):
        """Initialize the drudge object."""

        # Initialize the base su2 problem.
        super().__init__(ctx, **kwargs)

        # Set the range and dummies.
        # self.add_resolver_for_dumms()

        self.all_orb_range = all_orb_range
        self.all_orb_dumms = tuple(all_orb_dumms)
        self.set_dumms(all_orb_range, all_orb_dumms)
        self.set_name(*self.all_orb_dumms)
        self.add_resolver({
            i: (all_orb_range) for i in all_orb_dumms
        })

        # Set the operator attributes

        self.uga = uga

        # Make additional name definition for the operators.
        self.set_name(**{
            uga.label[0]+'_':uga,
            'eps':energies,
            'G':interact
        })
        
        #uga generaters can also be called using just 'uga[p,q]' instead of 'E_[p,q]'
        self.set_name(uga) 
        
        # Defining spec for passing to an external function - the Swapper
        spec = _UGSpec(
                uga=uga,
        )
        self._spec = spec
        
        # Create an instance of the ProjectedBCS or AGP Drudge to map from E_pq
        # to D_dag, N, D so that normal ordering and vev evaluation can be done
        self.eta = _eta
        self.sigma = _sigma
        self.set_name(**{
            'eta':self.eta,
            'sigma':self.sigma
        })

        self._agp_dr = ProjectedBCSDrudge(
            ctx, all_orb_range=self.all_orb_range, all_orb_dumms=self.all_orb_dumms
        )

        """Mapping E_pq to D_dag, N, D"""
        D_p = self._agp_dr.raise_
        N_ = self._agp_dr.cartan
        D_m = self._agp_dr.lower

        self.D_p = D_p
        self.N_ = N_
        self.D_m = D_m

        gen_idx1, gen_idx2 = self.all_orb_dumms[:2]

        epq_def = self.define(
            uga,gen_idx1,gen_idx2,
            self._agp_dr.sigma[gen_idx1,gen_idx2]*( self.eta[gen_idx1]*D_m[gen_idx1,gen_idx2] + \
                self.eta[gen_idx2]*D_p[gen_idx1,gen_idx2] ) + \
                KroneckerDelta(gen_idx1,gen_idx2)*N_[gen_idx1]
        )

        #Tensor Definitions for uga2agp
        self._uga2agp_defs = [
            epq_def
        ]

        """Mapping D_dag, N, D to E_pq"""
        dm_pq_def = self.define(
            D_m, gen_idx1, gen_idx2,
            (self.eta[gen_idx1]*uga[gen_idx1,gen_idx2] - self.eta[gen_idx2]*uga[gen_idx2,gen_idx1])
        )

        dp_pq_def = self.define(
            D_p, gen_idx1, gen_idx2,
            (self.eta[gen_idx1]*uga[gen_idx2,gen_idx1] - self.eta[gen_idx2]*uga[gen_idx1,gen_idx2])
        )

        np_def = self.define(
            N_, gen_idx1, uga[gen_idx1,gen_idx1]
        )
        
        #Tensor definitions for agp2uga
        self._agp2uga_defs = [
            dm_pq_def,
            dp_pq_def,
            np_def
        ]

        # set the Swapper
        self._swapper = functools.partial(_swap_ug, spec=spec)


    @property
    def swapper(self) -> GenQuadDrudge.Swapper:
        """Swapper for the new AGP Algebra."""
        return self._swapper

    def _isUGA(self,term):
        """Function returns True if the vectors are uga generators, otherwise False"""
        vecs = term.vecs
        for v in vecs:
            if v.base!=self.uga:
                raise ValueError('Unexpected generator of the Unitary Group',v)
        return [Term(sums=term.sums, amp=term.amp, vecs=term.vecs)]

    def _transl2agp(self, tensor: Tensor):
        """Translate a tensor object in terms of the fermion operators.

        This is an internal utility.  The resulting tensor has the internal
        AGP drudge object as its owner.
        """
        chk = tensor.bind(self._isUGA)

        # for t in trms:
        #     if ~_isUGA(t):
        #         raise ValueError('Unexpected generator of the Unitary Group',vec)

        return Tensor(
            self._agp_dr,
            tensor.subst_all(self._uga2agp_defs).terms
        )

    def _transl2uga(self, tensor: Tensor):
        """Translate a tensor object in terms of the fermion operators.

        This is an internal utility.  The resulting tensor has the internal
        AGP drudge object as its owner.
        """
        chk = tensor.bind(self._agp_dr._isAGP)

        # for t in trms:
        #     if ~self._agp_dr._isAGP(t):
        #         raise ValueError('Unexpected generator of the AGP Algebra',vec)

        return Tensor(
            self,
            tensor.subst_all(self._agp2uga_defs).terms
        )

    def get_vev_agp(self,h_tsr: Tensor,):
        """Function to evaluate the expectation value of a normal
        ordered tensor 'h_tsr' with respect to the projected BCS
        or the AGP state. This can be done after translating the the
        unitary group terms into the AGP basis algebra.
            h_tsr = tensor whose VEV is to be evaluated
        """
        transled = self._transl2agp(h_tsr)
        transled = self._agp_dr.agp_simplify(transled,final_step=True)
        res = self._agp_dr.get_vev(transled)
        return Tensor(self, res.terms)

    # def clean_beta(self, tnsr: Tensor):
    #     return tnsr.filter(_non_zero_by_beta)



_UGSpec = collections.namedtuple('_UGSpec',[
    'uga'
])


def _swap_ug(vec1: Vec, vec2: Vec, depth=None, *,spec: _UGSpec):
    """Swap two vectors based on the commutation rules for Unitary Group generators.
    Here, we introduce an additional input parameter 'depth' which is never
    specified by the user. Rather, it is put to make use os the anti-symmetric 
    nature of the commutation relations and make the function def compact. 
    """
    if depth is None:
        depth = 1
    
    char1, indice1, key1 = _parse_vec_uga(vec1,spec)
    char2, indice2, key2 = _parse_vec_uga(vec2,spec)
    
    p = indice1[0]
    q = indice1[1]
    r = indice2[0]
    s = indice2[1]

    if p != r:
        if p is min(p,r,key=default_sort_key):
            return None
        elif r is min(p,r,key=default_sort_key):
            expr = KroneckerDelta(q,r)*spec.uga[p,s] - KroneckerDelta(p,s)*spec.uga[r,q]
            return _UNITY, expr
        else:
            return None
    elif p == r:
        if q is min(q,s,key=default_sort_key):
            return None
        elif s is min(q,s,key=default_sort_key):
            expr = KroneckerDelta(q,r)*spec.uga[p,s] - KroneckerDelta(p,s)*spec.uga[r,q]
            return _UNITY, expr
        else:
            return None
    else:
        assert False




def _parse_vec_uga(vec, spec: _UGSpec):
    """Get the character, lattice indices, and the indices of keys of vector.
    """
    base = vec.base
    if base == spec.uga:
        char = _UG_GEN
    else:
        raise ValueError('Unexpected generator of the Unitary Group',vec)
    
    indices = vec.indices
    if len(indices)!=2:
        raise ValueError('Unexpected length of indices for the generators of Unitary Group',vec)
    keys = tuple(sympy_key(i) for i in indices)

    return char, indices, keys



_AGPSpec = collections.namedtuple('_AGPSpec',[
    'cartan',
    'raise_',
    'lower',
    'eta',
    'sigma'
])


def _swap_agp(vec1: Vec, vec2: Vec, depth=None, *,spec: _AGPSpec):
    """Swap two vectors based on the AGP operators commutation rules
    Here, we introduce an additional input parameter 'depth' which is never
    specified by the user. Rather, it is put to make use os the anti-symmetric 
    nature of the commutation relations and make the function def compact. 
    """
    if depth is None:
        depth = 1
    
    char1, indice1, key1 = _parse_vec_agp(vec1,spec)
    char2, indice2, key2 = _parse_vec_agp(vec2,spec)
    
    isCartan1 = (char1==1) and (len(indice1)==1)
    isCartan2 = (char2==1) and (len(indice2)==1)

    notCartan1 = (char1!=1) and (len(indice1)==2)
    notCartan2 = (char2!=1) and (len(indice2)==2)

    if not((isCartan1 or notCartan1) and (isCartan2 or notCartan2)):
        raise ValueError(
            'Invalid AGP generators on lattice', (vec1, vec2),
            'Inappropriate rank of indices with the input operator'
        )

    eta = spec.eta
    beta = spec.sigma

    if char1 == _RAISE:
    
        return None
    
    elif char1 == _CARTAN:

        r = indice1[0]

        if char2 == _RAISE:
            p = indice2[0]
            q = indice2[1]
            del_rq = KroneckerDelta(q,r)
            del_rp = KroneckerDelta(p,r)
            b_pq = beta[p,q]
            b_qp = beta[q,p]

            expr = -(del_rp - del_rq)*(eta[p]*eta[p]*b_pq - eta[q]*eta[q]*b_qp)*spec.raise_[p,q]\
                + (del_rp - del_rq)*eta[p]*eta[q]*(b_pq - b_qp)*spec.lower[p,q]

            return _UNITY, expr

        else:

            return None

    elif char1 == _LOWER:

        p = indice1[0]
        q = indice1[1]

        if char2 == _RAISE:
            r = indice2[0]
            s = indice2[1]
            
            def D_Ddag_comm_expr(a,b,c,d):
                del_ac = KroneckerDelta(a,c)
                del_bd = KroneckerDelta(b,d)
                b_ac = beta[a,c]
                b_db = beta[d,b]

                exprn = del_ac*del_bd*(spec.cartan[a]-spec.cartan[b]) + \
                    del_bd*b_ac*( eta[a]*spec.lower[a,c] - eta[c]*spec.raise_[a,c]) +\
                    del_ac*b_db*( eta[d]*spec.lower[b,d] - eta[b]*spec.raise_[b,d])
                exprn = eta[a]*eta[c]*exprn
                return exprn
            expr1 = D_Ddag_comm_expr(p,q,r,s)
            expr2 = D_Ddag_comm_expr(q,p,s,r)
            expr3 = -D_Ddag_comm_expr(p,q,s,r)
            expr4 = -D_Ddag_comm_expr(q,p,r,s)

            tot_comm = (expr1 + expr2 + expr3 + expr4)

            return _UNITY, tot_comm

        elif char2 == _CARTAN:
            r = indice2[0]
            del_rq = KroneckerDelta(q,r)
            del_rp = KroneckerDelta(p,r)
            b_pq = beta[p,q]
            b_qp = beta[q,p]

            expr = -(eta[p]*eta[p]*b_pq - eta[q]*eta[q]*b_qp)*spec.lower[p,q]
            expr += eta[p]*eta[q]*(b_pq - b_qp)*spec.raise_[p,q]
            # expr *= (del_rp - del_rq)
            
            return _UNITY, expr*(del_rp - del_rq)

        else:

            return None

    else:
        assert False


_UG_GEN = 0
_RAISE = 0
_CARTAN = 1
_LOWER = 2

_UNITY = Integer(1)
_NOUGHT = Integer(0)
_TWO = Integer(2)

_eta = IndexedBase(r'\eta')
_sigma = IndexedBase(r'\sigma')

def _parse_vec_agp(vec, spec: _AGPSpec):
    """Get the character, lattice indices, and the indices of keys of vector.
    """
    base = vec.base
    if base == spec.cartan:
        char = _CARTAN
    elif base == spec.raise_:
        char = _RAISE
    elif base == spec.lower:
        char = _LOWER
    else:
        raise ValueError('Unexpected vector for the AGP algebra',vec)
    
    indices = vec.indices
    keys = tuple(sympy_key(i) for i in indices)

    return char, indices, keys

# def _non_zero_by_beta(term: Term):
#     """Test if a term is not zero symmetry of beta.
#     """
# 
#     amp = term.amp
#     amp_str = srepr(amp)
#     bet_list = [m.end() for m in re.finditer('beta',amp_str)]
# 
#     if len(bet_list):
#         for ind in bet_list:
#             ind1 = Symbol(amp_str[ind+13])
#             ind2 = Symbol(amp_str[ind+13+13])
#             if ind1 == ind2:
#                 return False
#             else:
#                 continue
# 
#     return True
