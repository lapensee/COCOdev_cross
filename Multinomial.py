#obtained from http://stackoverflow.com/questions/13903922/multinomial-pmf-in-python-scipy-numpy

import math
import numpy as np #usage of np.log instead - makes the function vectorizable

class Multinomial(object):
  def __init__(self, params):
    self._params = params

  def pmf(self, counts):
    if not(len(counts)==len(self._params)):
      raise ValueError("Dimensionality of count vector is incorrect")

    prob = 1.
    for i,c in enumerate(counts):
      prob *= self._params[i]**counts[i]

    return prob * np.exp(self._log_multinomial_coeff(counts))

  def log_pmf(self,counts):
    if not(len(counts)==len(self._params)):
      raise ValueError("Dimensionality of count vector is incorrect")

    prob = 0.
    for i,c in enumerate(counts):
      prob += counts[i]*np.log(self._params[i])

    return prob + self._log_multinomial_coeff(counts)

  def _log_multinomial_coeff(self, counts):
    return self._log_factorial(sum(counts)) - sum(self._log_factorial(c)
                                                    for c in counts)

  def _log_factorial(self, num):
    if not round(num)==num and num > 0:
      raise ValueError("Can only compute the factorial of positive ints")
    return np.sum(np.log(n) for n in range(1,num+1))
