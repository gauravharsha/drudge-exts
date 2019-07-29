"""
Configuration file for SU(4) Lipkin Model in the
Author: Gaurav Harsha
Date: July 29, 2019
"""

import collections
import functools

from dummy_spark import SparkContext
# from pyspark import SparkContext
from sympy import Symbol, collect, Add, Mul, Integer, symbols, factor, diff
from su4 import *


ctx = SparkContext('local[*]','su4')

dr = SU4LatticeDrudge(ctx)
nams = dr.names

DRUDGE = dr
