"""Configure a drudge for Projected BCS basis."""

from dummy_spark import SparkContext
# from pyspark import SparkContext
from sympy import Symbol, collect, Add, Mul, Integer, symbols, factor, diff, IndexedBase
from drudge import Perm, IDENT, NEG
from ugagp import *
import pdb

ctx = SparkContext()
dr = UnitaryGroupDrudge(ctx)

G = Symbol('G')
A = dr.all_orb_range
dr.add_default_resolver(dr.all_orb_range)

dr.set_name(G, A)

# pnames = dr.names

dr.set_name(
    Perm=Perm,
    IDENT=IDENT,
    NEG=NEG,
    pdb=pdb
)

DRUDGE = dr
