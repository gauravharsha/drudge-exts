"""Configures a simple drudge for reduced BCS model."""

from dummy_spark import SparkContext
#from pyspark import SparkContext

from sympy import Symbol, collect, Add, Mul, Integer, symbols, factor, diff, IndexedBase
from agp_fermi import *
from drudge import InvariantIndexable, Perm, IDENT, NEG

ctx = SparkContext()
dr = AGPFermi(ctx)

#==================================== 
# AGP expected values:
#====================================
# case: z00 
Z00 = IndexedBase('Z00')

# case: z02 
Z02 = IndexedBase('Z02')
dr.set_symm(Z02,
    Perm([1,0],IDENT),
    valence=2
)

# case: z04
Z04 = IndexedBase('Z04')
dr.set_symm(Z04,
    Perm([1,0,2,3],IDENT),
    Perm([0,1,3,2],IDENT),
    Perm([2,3,0,1],IDENT),
    valence = 4
)

# case: z06
Z06 = IndexedBase('Z06')
dr.set_symm(Z06,
    Perm([1,0,2,3,4,5],IDENT),
    Perm([0,2,1,3,4,5],IDENT),
    Perm([0,1,2,4,3,5],IDENT),
    Perm([0,1,2,3,5,4],IDENT),
    Perm([3,4,5,0,1,2],IDENT),
    valence=6
)

# case: z08
Z08 = IndexedBase('Z08')
dr.set_symm(Z08,
    Perm([1,0,2,3,4,5,6,7],IDENT),
    Perm([0,2,1,3,4,5,6,7],IDENT),
    Perm([0,1,3,2,4,5,6,7],IDENT),
    Perm([0,1,2,3,5,4,6,7],IDENT),
    Perm([0,1,2,3,4,6,5,7],IDENT),
    Perm([0,1,2,3,4,5,7,6],IDENT),
    Perm([4,5,6,7,0,1,2,3],IDENT),
    valence=8
)

#--------------------------------------
# case: z11
Z11 = IndexedBase('Z11')

# case: z13
Z13 = IndexedBase('Z13')
dr.set_symm(Z13,
    Perm([2,1,0],IDENT),
    valence=3
) 

# case: z15
Z15 = IndexedBase('Z15')
dr.set_symm(Z15,
    Perm([1,0,2,3,4],IDENT),
    Perm([0,1,2,4,3],IDENT),
    Perm([3,4,2,0,1],IDENT),
    valence=5
)

# case: z17
Z17 = IndexedBase('Z17')
dr.set_symm(Z17,
    Perm([1,0,2,3,4,5,6],IDENT),
    Perm([0,2,1,3,4,5,6],IDENT),
    Perm([0,1,2,3,5,4,6],IDENT),
    Perm([0,1,2,3,4,6,5],IDENT),
    Perm([4,5,6,3,0,1,2],IDENT),
    valence=7
)

#--------------------------------------
# case: z22
Z22 = IndexedBase('Z22')
dr.set_symm(Z22,
    Perm([1,0],IDENT),
    valence=2
) 

# case: z24
Z24 = IndexedBase('Z24')
dr.set_symm(Z24,
    Perm([3,1,2,0],IDENT),
    Perm([0,2,1,3],IDENT),
    valence=4
) 

# case: z26
Z26 = IndexedBase('Z26')
dr.set_symm(Z26,
    Perm([0,1,3,2,4,5],IDENT),
    Perm([1,0,2,3,4,5],IDENT),
    Perm([0,1,2,3,5,4],IDENT),
    Perm([4,5,2,3,0,1],IDENT),
    valence=6
) 

#--------------------------------------
#case:z33
Z33 = IndexedBase('Z33')
dr.set_symm(Z33,
    Perm([1,0,2],IDENT),
    Perm([0,2,1],IDENT),
    valence=3
)

# case: z35
Z35 = IndexedBase('Z35')
dr.set_symm(Z35,
    Perm([0,2,1,3,4],IDENT),
    Perm([0,1,3,2,4],IDENT),
    Perm([4,1,2,3,0],IDENT),
    valence=5
) 

#--------------------------------------
# case: z44
Z44 = IndexedBase('Z44')
dr.set_symm(Z44,
    Perm([1,0,2,3],IDENT),
    Perm([0,2,1,3],IDENT),
    Perm([0,1,3,2],IDENT),
    valence=4
) 

rdm_list = [[Z00, Z02, Z04, Z06, Z08], [Z11, Z13, Z15, Z17], [Z22, Z24, Z26], [Z33, Z35], [Z44]]

dr.set_name(
    Symbol=Symbol,
    symbols=symbols,
    rdm_list=rdm_list,
)

DRUDGE = dr
