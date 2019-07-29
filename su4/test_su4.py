"""
   Python Script for Testing some basic commutation relations in the SU(4) drudge
"""

from sympy import *
from drudge import *
from su4 import *
from dummy_spark import SparkContext

ctx = SparkContext()
dr = SU4LatticeDrudge(ctx)

names = dr.names

J_p = names.J_p
J_ = names.J_
J_m = names.J_m

K_p = names.K_p
K_ = names.K_
K_m = names.K_m

Y_pp = names.Y_pp
Y_pm = names.Y_pm
Y_pz = names.Y_pz

Y_mp = names.Y_mp
Y_mm = names.Y_mm
Y_mz = names.Y_mz

Y_zp = names.Y_zp
Y_zm = names.Y_zm
Y_zz = names.Y_zz

# Defining the tensor form of the operators
jp = dr.sum(J_p)
jz = dr.sum(J_)
jm = dr.sum(J_m)

kp = dr.sum(K_p)
kz = dr.sum(K_)
km = dr.sum(K_m)

ypp = dr.sum(Y_pp)
ymm = dr.sum(Y_mm)
yzz = dr.sum(Y_zz)

ypm = dr.sum(Y_pm)
ymp = dr.sum(Y_mp)
ypz = dr.sum(Y_pz)
yzp = dr.sum(Y_zp)
ymz = dr.sum(Y_mz)
yzm = dr.sum(Y_zm)

# Test results
print( dr.simplify(jp | jm) == Integer(2)*jz)
print( dr.simplify(jz | jp) == jp)
print( dr.simplify(jz | jm) == Integer(-1)*jm)

print( dr.simplify(kp | km) == Integer(2)*kz)
print( dr.simplify(kz | kp) == kp)
print( dr.simplify(kz | km) == Integer(-1)*km)
