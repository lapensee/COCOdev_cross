def hpd(data, level = .95) :
  """ The Highest Posterior Density (credible) interval of data at level level.

  :param data: sequence of real values
  :param level: (0 < level < 1)
  """ 
  
  d = list(data)
  d.sort()

  nData = len(data)
  nIn = int(round(level * nData))
  if nIn < 2 :
    raise RuntimeError("not enough data")
  
  i = 0
  r = d[i+nIn-1] - d[i]
  for k in range(len(d) - (nIn - 1)) :
    rk = d[k+nIn-1] - d[k]
    if rk < r :
      r = rk
      i = k

  assert 0 <= i <= i+nIn-1 < len(d)
  
  return (d[i], d[i+(nIn/2)-1],d[i+nIn-1])

import numpy as np
def HDI_from_MCMC(posterior_samples, credible_mass):
  # Computes highest density interval from a sample of representative values,
  # estimated as the shortest credible interval
  # Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
  sorted_points = sorted(posterior_samples)
  ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
  nCIs = len(sorted_points) - ciIdxInc
  ciWidth = [0]*nCIs
  for i in range(0, nCIs): ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
  HDImin = sorted_points[ciWidth.index(min(ciWidth))]
  HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
  return(HDImin, HDImax)
