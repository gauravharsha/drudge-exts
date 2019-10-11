"""Configures a simple drudge for reduced BCS model."""

from dummy_spark import SparkContext
#from pyspark import SparkContext

from sympy import Symbol, collect, Add, Mul, Integer, symbols, factor, diff, IndexedBase
from agp_fermi import *
from drudge import InvariantIndexable, Perm, IDENT, NEG

ctx = SparkContext()
dr = AGPFermi(ctx)

dr.set_name(
    Symbol=Symbol,
    symbols=symbols
)

DRUDGE = dr
