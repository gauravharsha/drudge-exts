from dummy_spark import SparkContext
from drudge import *
from works import *

# Define few shortcuts
DN = DOWN
delK = KroneckerDelta

def test_fermi_anti_comm_rules():

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
    N_q = names.N[q]

    # Commutation relations
    expr1 = dr.simplify(Pdag_p * P_q - P_q * Pdag_p)
    expr2 = dr.simplify(N_q * Pdag_p - Pdag_p * N_q)
    expr3 = dr.simplify(N_p * P_q - P_q * N_p)

    # Assertions
    assert dr.simplify(expr1 - delK(p, q)*(names.N[p] - 1)) == 0
    assert dr.simplify(expr2 - 2*delK(p, q)*Pdag_p) == 0
    assert dr.simplify(expr3 + 2*delK(p, q)*P_q) == 0

