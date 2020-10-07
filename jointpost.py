from extract import *
from plotFit import violinPlot
from pylab import *
from scipy.stats import linregress
import os
from copy import deepcopy
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

version='Vaa_rBeta'
fits = ['sourceDev4COCOfreq' + version, 'sourceDev8COCOfreq' + version, 'sourceDevAdultsCOCOfreq' + version]
p = getParams(fits[0])
labels = ['4-5yrs', '7-8yrs', 'Adults']

graphs = []
for f, fit in enumerate(fits):
  print f
  phiMu = getPhiMu(fit)
  p = getParams(fit)
  pp = [ 'logVti','logVsu','logVaa', 'logVac']

  val1 = phiMu[:,:,p.index('logVti')]
  val2 = phiMu[:,:,p.index('logVsu')]
  val3 = phiMu[:,:,p.index('logVaa')]
  val4 = phiMu[:,:,p.index('logVac')]

  df = pd.DataFrame({'logVti': val1.flatten()})
  df['logVsu'] = val2.flatten()
  df['logVaa'] = val3.flatten()
  df['logVac'] = val4.flatten()
  #pd.scatter_matrix(df, figsize=(6, 6))
  pd.plotting.scatter_matrix(df, alpha=0.5, figsize=(10, 10))#, diagonal='kde')
  plt.suptitle(labels[f])
  plt.savefig(labels[f]+'.png')
  #plt.show()
  #sns_plot.savefig(labels[f]+'.pdf')

