from dummy_spark import SparkContext
from drudge import *
from agp_fermi import *

# Define few shortcuts
DN = DOWN
delK = KroneckerDelta

def test_fermi_anti_comm_rules():

    # Test commutation relations for the fermionic algebra

    # Initialise drudge
    ctx = SparkContext()
    dr = AGPFermi(ctx)

    # namespace
    names = dr.names

    # Indices
    p, q = names.A_dumms[:2]

    # fermion operators
    cdag_p_up = names.c_dag[p, UP]
    cdag_p_dn = names.c_dag[p, DN]
    c_q_up = names.c_[q, UP]
    c_q_dn = names.c_[q, DN]

    # Anti-commutation relations
    expr1 = dr.simplify(cdag_p_up * c_q_dn + c_q_dn * cdag_p_up)
    expr2 = dr.simplify(cdag_p_dn * c_q_up + c_q_up * cdag_p_dn)
    expr3 = dr.simplify(cdag_p_up * c_q_up + c_q_up * cdag_p_up)
    expr4 = dr.simplify(cdag_p_dn * c_q_dn + c_q_dn * cdag_p_dn)

    # Assertions
    assert expr1 == 0
    assert expr2 == 0
    assert dr.simplify(expr3 - delK(p, q)) == 0
    assert dr.simplify(expr4 - delK(p, q)) == 0

def test_pairing_comm_rules():

    # Test commutation relations for the pairing SU2 algebra

    # Initialise drudge
    ctx = SparkContext()
    dr = AGPFermi(ctx)

    # namespace
    names = dr.names

    # Indices
    p, q = names.A_dumms[:2]

    # BCS Operators
    Pdag_p = names.P_dag[p]
    P_q = names.P_[q]
    N_p = names.N[p]
    Nup_p = dr.N_up[p]
    Ndn_p = dr.N_dn[p]
    N_q = names.N[q]
    Nup_q = dr.N_up[q]
    Ndn_q = dr.N_dn[q]

    # Commutation relations
    expr1 = dr.simplify(Pdag_p * P_q - P_q * Pdag_p)
    expr2 = dr.simplify(N_q * Pdag_p - Pdag_p * N_q)
    expr2a = dr.simplify(Nup_q * Pdag_p - Pdag_p * Nup_q)
    expr2b = dr.simplify(Ndn_q * Pdag_p - Pdag_p * Ndn_q)
    expr3 = dr.simplify(N_p * P_q - P_q * N_p)
    expr3a = dr.simplify(Nup_p * P_q - P_q * Nup_p)
    expr3b = dr.simplify(Ndn_p * P_q - P_q * Ndn_p)

    # Assertions
    assert dr.simplify(expr1 - delK(p, q)*(names.N[p] - 1)) == 0
    assert dr.simplify(expr2 - 2*delK(p, q)*Pdag_p) == 0
    assert dr.simplify(expr2a - delK(p, q)*Pdag_p) == 0
    assert dr.simplify(expr2b - delK(p, q)*Pdag_p) == 0
    assert dr.simplify(expr3 + 2*delK(p, q)*P_q) == 0
    assert dr.simplify(expr3a + delK(p, q)*P_q) == 0
    assert dr.simplify(expr3b + delK(p, q)*P_q) == 0

def test_spinflip_su2_comm_rules():

    # Test commutation relations for the spin-flip SU2 algebra

    # Initialise drudge
    ctx = SparkContext()
    dr = AGPFermi(ctx)

    # namespace
    names = dr.names

    # Indices
    p, q = names.A_dumms[:2]

    # BCS Operators
    Jp_p = names.J_p[p]
    Jm_q = names.J_m[q]
    Jz_p = names.J_z[p]
    Jz_q = names.J_z[q]

    # Commutation relations
    expr1 = dr.simplify(Jp_p * Jm_q - Jm_q * Jp_p)
    expr2 = dr.simplify(Jz_q * Jp_p - Jp_p * Jz_q)
    expr3 = dr.simplify(Jz_p * Jm_q - Jm_q * Jz_p)

    # Assertions
    assert dr.simplify(expr1 - delK(p, q) * 2 * Jz_p) == 0
    assert dr.simplify(expr2 - delK(p, q) * Jp_p) == 0
    assert dr.simplify(expr3 + delK(p, q) * Jm_q) == 0

def test_inter_algebra_comm_rules():

    # Test commutation rules between the fermionic and the two SU2 algebras

    # Initialise drudge
    ctx = SparkContext()
    dr = AGPFermi(ctx)

    # namespace
    names = dr.names

    # Indices
    p, q, r, s = names.A_dumms[:4]

    # Operators
    cdag_p = names.c_dag[p, UP]
    c_p = names.c_[p, UP]
    
    N_q = names.N_[q]
    Pdag_q = names.P_dag[q]
    P_q = names.P_[q]

    Jp_r = names.J_p[r]
    Jm_r = names.J_m[r]
    Jz_r = names.J_z[r]

    # Commutation rules
    expr1a = dr.simplify(N_q * cdag_p - cdag_p * N_q)
    expr1b = dr.simplify(N_q * c_p - c_p * N_q)
    expr1c = dr.simplify(Pdag_q * cdag_p - cdag_p * Pdag_q)
    expr1d = dr.simplify(Pdag_q * c_p - c_p * Pdag_q)
    expr1e = dr.simplify(P_q * cdag_p - cdag_p * P_q)
    expr1f = dr.simplify(P_q * c_p - c_p * P_q)

    expr2a = dr.simplify(N_q * Jp_r - Jp_r * N_q)
    expr2b = dr.simplify(N_q * Jm_r - Jm_r * N_q)
    expr2c = dr.simplify(N_q * Jz_r - Jz_r * N_q)

    expr3a = dr.simplify(P_q * Jp_r - Jp_r * P_q)
    expr3b = dr.simplify(P_q * Jm_r - Jm_r * P_q)
    expr3c = dr.simplify(P_q * Jz_r - Jz_r * P_q)

    expr4a = dr.simplify(Pdag_q * Jp_r - Jp_r * Pdag_q)
    expr4b = dr.simplify(Pdag_q * Jm_r - Jm_r * Pdag_q)
    expr4c = dr.simplify(Pdag_q * Jz_r - Jz_r * Pdag_q)

    expr5a = dr.simplify(Jz_r * cdag_p - cdag_p * Jz_r)
    expr5b = dr.simplify(Jz_r * c_p - c_p * Jz_r)
    expr5c = dr.simplify(Jp_r * cdag_p - cdag_p * Jp_r)
    expr5d = dr.simplify(Jp_r * c_p - c_p * Jp_r)
    expr5e = dr.simplify(Jm_r * cdag_p - cdag_p * Jm_r)
    expr5f = dr.simplify(Jm_r * c_p - c_p * Jm_r)

    # Assertions
    assert dr.simplify(expr1a - delK(p, q)*cdag_p) == 0
    assert dr.simplify(expr1b + delK(p, q)*c_p) == 0
    assert expr1c == 0
    assert dr.simplify(expr1d + delK(p, q)*names.c_dag[p, DN]) == 0
    assert dr.simplify(expr1e - delK(p, q)*names.c_[p, DN]) == 0
    assert expr1f == 0

    assert expr2a == 0
    assert expr2b == 0
    assert expr2c == 0

    assert expr3a == 0
    assert expr3b == 0
    assert expr3c == 0

    assert expr4a == 0
    assert expr4b == 0
    assert expr4c == 0

    assert dr.simplify(expr5a - delK(p, r)*cdag_p/2) == 0
    assert dr.simplify(expr5b + delK(p, r)*c_p/2) == 0
    assert expr5c == 0
    assert dr.simplify(expr5d + delK(p, r)*names.c_[p, DN]) == 0
    assert dr.simplify(expr5e - delK(p, r)*names.c_dag[p, DN]) == 0
    assert expr5f == 0

def test_nilpotency_of_operators():

    # Test the nilpotency of fermi and Pairing-SU2 operators

    # Initialise drudge
    ctx = SparkContext()
    dr = AGPFermi(ctx)

    # namespace
    names = dr.names

    # Indices
    p, q, r = names.A_dumms[:3]

    # Operators
    cdag_p = names.c_dag[p, UP]
    c_p = names.c_[p, UP]
    
    N_q = names.N_[q]
    Pdag_q = names.P_dag[q]
    P_q = names.P_[q]

    Jp_r = names.J_p[r]
    Jm_r = names.J_m[r]
    Jz_r = names.J_z[r]

    # Expressions
    expr1a = dr.simplify(cdag_p * cdag_p)
    expr1b = dr.simplify(c_p * c_p)

    expr2a = dr.simplify(Pdag_q * Pdag_q)
    expr2b = dr.simplify(P_q * P_q)

    expr3a = dr.simplify(Jp_r * Jp_r)
    expr3b = dr.simplify(Jm_r * Jm_r)

    # assertions
    assert expr1a == 0
    assert expr1b == 0

    assert expr2a == 0
    assert expr2b == 0

    assert expr3a == 0
    assert expr3b == 0

def test_nonzero_by_cartan():

    # For pairing algebra, N_p * P_p or Pdag_p * N_p should be ZERO
    #   That is what we test here

    # Initialise drudge
    ctx = SparkContext()
    dr = AGPFermi(ctx)

    # namespace
    names = dr.names

    # Indices
    p = names.A_dumms[0]

    # Operators
    N_p = names.N_[p]
    Pdag_p = names.P_dag[p]
    P_p = names.P_[p]

    # expressions
    expr1 = dr.simplify(Pdag_p * N_p)
    expr2 = dr.simplify(N_p * P_p)

    # assertions
    assert expr1 == 0
    assert expr2 == 0

def test_unique_indices_functionality():

    # Test for unique indices functionality

    # Initialise drudge
    ctx = SparkContext()
    dr = AGPFermi(ctx)

    # namespace
    names = dr.names

    # Indices
    p, q, r, s = names.A_dumms[:4]

    # list of unique indices shoule be empty
    assert dr.unique_del_lists == []

    # declare r and s to be unique indices
    dr.unique_indices([r, s])

    # check unique indices list now
    # Basically, unique ind list is a list of tuples
    assert dr.unique_del_lists[0] == {r, s}

    # Expression evaluation
    e_pq = names.e_[p, q]
    expr = dr.simplify(
        (delK(r, s) + delK(p, r)) * e_pq
    )
    expr2 = dr.simplify(
        delK(p, r) * e_pq
    )

    # assertion
    assert dr.simplify(expr - expr2) == 0

def test_canonical_ordering():

    # Test the canonical ordering functionality

    # Initialise drudge
    ctx = SparkContext()
    dr = AGPFermi(ctx)

    # namespace
    names = dr.names

    # Indices
    p, q, r, s = names.A_dumms[:4]

    # Operators
    cdag_p_up = names.c_dag[p, UP]
    cdag_p_dn = names.c_dag[p, DN]
    c_q_up = names.c_[q, UP]
    c_q_dn = names.c_[q, DN]
    
    Pdag_p = names.P_dag[p]
    N_q = names.N_[q]
    P_r = names.P_[r]

    Jp_p = names.J_p[p]
    Jz_q = names.J_z[q]
    Jm_r = names.J_m[r]

    # Let all the indices be unique - so no commutation terms arise
    dr.unique_indices([p, q, r, s])

    # expression for intra algebra ordering
    expr1 = dr.simplify(c_q_up * c_q_dn * cdag_p_up * cdag_p_dn)
    expr2 = dr.simplify(P_r * N_q * Pdag_p)
    expr3 = dr.simplify(Jm_r * Jz_q * Jp_p)

    # assertions
    assert dr.simplify(expr1 + cdag_p_up * cdag_p_dn * c_q_dn * c_q_up) == 0
    assert dr.simplify(expr2 - Pdag_p * N_q * P_r) == 0
    assert dr.simplify(expr3 - Jp_p * Jz_q * Jm_r) == 0

    # expressions for inter algebra ordering
    Pdag_r = names.P_dag[r]
    N_r = names.N_[r]
    expr1a = dr.simplify(cdag_p_up * cdag_p_dn * Pdag_r * N_r * P_r)

    Jp_q = names.J_p[q]
    Jm_q = names.J_m[q]
    expr2a = dr.simplify(cdag_p_up * cdag_p_dn * Pdag_r * Jp_q * Jz_q * Jm_q)

    # assertions
    assert dr.simplify(expr1a - Pdag_r * N_r * P_r * cdag_p_up * cdag_p_dn) == 0
    assert dr.simplify(expr2a - Pdag_r * Jp_q * Jz_q * Jm_q * cdag_p_up * cdag_p_dn) == 0

def test_get_seniority_zero():

    # Get seniority zero expressions corresponding to some test results that we know already
    #   This will indirectly also include testing of extract_su2

    # Initialise drudge
    ctx = SparkContext()
    dr = AGPFermi(ctx)

    # namespace
    names = dr.names

    # Indices
    p, q, r, s = names.A_dumms[:4]

    # Operators
    cdag_p_up = names.c_dag[p, UP]
    cdag_p_dn = names.c_dag[p, DN]
    c_p_up = names.c_[p, UP]
    c_p_dn = names.c_[p, DN]

    # expression1: should simplify to Np Np /4
    expr1a = dr.simplify(cdag_p_up * cdag_p_dn * c_p_dn * c_p_up)
    expr1 = dr.get_seniority_zero(expr1a)
    res1 = dr.simplify(names.N_[p] * names.N_[p] / 4)

    # expression2: should simplify to 2 Pdag_p P_q (when p not= q)
    e_pq = names.e_[p, q]
    dr.unique_indices([p, q])
    expr2a = dr.simplify( e_pq * e_pq )
    expr2 = dr.get_seniority_zero(expr2a)
    res2 = dr.simplify(names.P_dag[p] * names.P_[q] * 2)

    # assertions
    assert dr.simplify(expr1 - res1) == 0
    assert dr.simplify(expr2 - res2) == 0
