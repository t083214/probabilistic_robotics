import sys
#sys.path.append('../scripts/')
from robot import *
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
import math
import numpy as np

def num(epsilon, delta, binnum):
    return math.ceil(chi2.ppf(1.0 - delta, binnum-1)/(2*epsilon))

fig, (axl, axr) = plt.subplots(ncols=2, figsize=(10, 4))
bs = np.arange(2, 10)
n = [num(0.1, 0.01, b) for b in bs]
axl.set_title("bin:2-10")
axl.plot(bs, n)
bs = np.arange(2, 100000)
n = [num(0.1, 0.01, b) for b in bs]
axr.set_title("bin:2-100000")
axr.plot(bs, n)

plt.draw()
plt.waitforbuttonpress(0)