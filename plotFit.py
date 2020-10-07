import cPickle as pickle
from pylab import *
import numpy as np
from numpy.lib import recfunctions
from de import dtnorm
from deFuncs import *
from scipy.stats import lognorm, gaussian_kde, gamma, norm, pearsonr
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from extract import *

ion()

legendFont = matplotlib.font_manager.FontProperties(size=10)
from hpd import hpd, HDI_from_MCMC

##### VIOLIN PLOT #####

def violinPlot(x, arr, scale = 1/75., nBins = 400, color = 'black', alpha = .92, linestyle = '-', linewidth = 1.5, returnGraph = False, cut = -2.50, dashes = (1,1), **kwargs):
  vals = arr.reshape(arr.size).astype(np.double)
  vals = vals[np.isfinite(vals)]
  vals = vals[arange(0, len(vals), 5)] #thinning
  kde = KDE(vals) #this doesn't seem to be working
  kde.fit(gridsize = nBins)
  incl = kde.density > 10**cut #density to cut - this keeps the tails from getting too long
  dens, support = kde.density[incl], kde.support[incl]
  #dens, support = kde.density, kde.support
  if linestyle != '-':
    a = plot(x - (dens * scale), support, color = color, linewidth = linewidth, linestyle = linestyle, dashes = dashes, alpha = alpha)
    plot(x + (dens * scale), support, color = color, linewidth = linewidth, linestyle = linestyle, dashes = dashes, alpha = alpha)
  else:
    a = plot(x - (dens * scale), support, color = color, linewidth = linewidth, linestyle = linestyle, alpha = alpha)
    plot(x + (dens * scale), support, color = color, linewidth = linewidth, linestyle = linestyle, alpha = alpha)
  if returnGraph == True: return a
  
def HDIbar(x, arr, marker = 'o', msize = 9, color = 'black', ecolor = 'black', p = .95, returnGraph = False, alpha = 1.0, ax = None):
  arr = arr.flatten()
  arr = arr[np.isfinite(arr)]
  lower, upper = HDI_from_MCMC(arr, p)
  m = np.mean(arr)
  if returnGraph == False:
    if ax == None: errorbar(x, m, yerr = np.array([m - lower, upper - m]).reshape(2,1), marker = marker, markersize = msize, ecolor = ecolor, color = color, alpha = alpha)
    else: ax.errorbar(x, m, yerr = np.array([m - lower, upper - m]).reshape(2,1), marker = marker, markersize = msize, ecolor = ecolor, color = color, alpha = alpha)
  else:
    if ax == None: return errorbar(x, m, yerr = np.array([m - lower, upper - m]).reshape(2,1), marker = marker, markersize = msize, ecolor = ecolor, color = color, alpha = alpha)[0]
    else: return ax.errorbar(x, m, yerr = np.array([m - lower, upper - m]).reshape(2,1), marker = marker, markersize = msize, ecolor = ecolor, color = color, alpha = alpha)[0]

def plotSVs(arr, params, sv = 'sv', color = 'black', burnin=1000, fontsize = 9):
  svs = []
  for param in params:
    if sv in param: svs.append(param)
  for i, sv in enumerate(svs): violinPlot(i + 1, arr[:,burnin:,params.index(sv)])
  axhline(1.0, color = 'black', linestyle = '--')
  xticks(arange(len(svs)) + 1, svs, fontsize = fontsize)
  yticks(fontsize = fontsize)

def plotParam(arr, target, params, color = 'black', burnin = 1000, fontsize = 10):
  ps = []
  for param in params:
    if target in param: ps.append(param)
  for i, param in enumerate(ps): violinPlot(i + 1, arr[:,burnin:,params.index(param)], color = color)
  xticks(arange(len(ps)) + 1, ps, fontsize = fontsize)
  yticks(fontsize = fontsize)
  
#use this function to check convergence of all parameter values
def subjConvergence(fit, cutoff = 1.05, cull = True):
  dataset = dataName(fit)
  params = getParams(fit)
  if dataset in ['AnnisSD', 'wprule', 'DR2', 'DR1words', 'DR1pics']: excludeSubj = True
  else: excludeSubj = False
  data = returnData(dataset, excludeSubj = excludeSubj)
  nSubj = len(np.unique(data['subj']))
  for subj in xrange(nSubj):
    theta = getSubj(subj, fit)
    nChains = theta.shape[0]
    if cull == True:
      theta = cullStuck(theta)
      if theta.shape[0] != nChains: print subj, nChains - theta.shape[0]
    for param in params:
      gr = gelman_rubin(theta[:,:,params.index(param)])
      #gr = pymc.diagnostics.gelman_rubin(theta[:,:,params.index(param)])
      if gr > cutoff: print 'Subject: ' + str(subj) + ' Parameter: ' + param + ' GR: ' + str(np.around(gr, 4))

def subjConvergenceBetween(fit, cutoff = 1.05):

  ps = getps(fit)
  params = getParams(fit)
  dataset = dataName(fit)
  data = returnData(dataset)
  nSubj = len(np.unique(data['subj']))
  nSubjs = [len(np.unique(data[data['exp'] == exp]['subj'])) for exp in [1,2,3]]
  phiMu = getPhiMu(fit)
  phiSigma = getPhiSigma(fit)
  
  #assemble thetas
  thetas = []
  c = 0
  for t in xrange(3):
    theta0 = getSubj(c, fit)
    theta = np.zeros((nSubjs[t], theta0.shape[0], theta0.shape[1], theta0.shape[2]))
    for i in xrange(nSubjs[t]):
      theta[i] = getSubj(i+c, fit)
    thetas.append(theta)
    c += nSubjs[t]

    x = 0
  print 'Subjects collected!'
    
  for g in xrange(len(thetas)):
    theta, p = thetas[g], ps[g]
    for param in p:
      for subj in xrange(theta.shape[0]):
        gr = gelman_rubin(theta[subj,:,:,p.index(param)])
        if gr > cutoff: print 'Subject: ' + str(subj + x) + ' Parameter: ' + param + ' GR: ' + str(np.around(gr, 4))
    x += theta.shape[0]

  for param in params:
    gr = gelman_rubin(phiMu[:,:,params.index(param)])
    if gr > cutoff: print 'phiMu Parameter: ' + param + ' GR: ' + str(np.around(gr, 4))
    gr = gelman_rubin(phiSigma[:,:,params.index(param)])
    if gr > cutoff: print 'phiSigma Parameter: ' + param + ' GR: ' + str(np.around(gr, 4))

def subjConvergence2(fit, dataset, cutoff = 1.07):
  params = getParams(fit)
  data = returnData(dataset)
  nSubj = len(np.unique(data['subj']))
  unmet = True
  i = 0

  theta0 = getSubj(0, fit)
  nChains, nmc, nParams = theta0.shape
  nSubj = len(np.unique(data['subj']))
  theta = np.zeros((nSubj,nChains,nmc,nParams))
  theta[0] = theta0
  for subj in xrange(1, nSubj): theta[subj] = getSubj(subj, fit)
  print 'Subjects collected!'
  while (unmet == True) & (i < 500):
    chains = range(90)
    shuffle(chains)
    chains = chains[0:3 * len(params)]
    m = 0
    pparam = ''
    unmet = False
    for param in params:
      gr = gelman_rubin(theta[8,chains,:,params.index(param)])
      if gr > cutoff: unmet = True
      if gr > m:
        m = gr
        pparam = param
    print m, pparam
  print 'Found the minimum... writing pickles...'
  phiMu = getPhiMu(fit)
  phiSigma = getPhiSigma(fit)
  weight = getWeight(fit)
  hyperWeight = getHyperWeight(fit)
  fit += '/'
  for subj in xrange(theta.shape[0]): pickle.dump(theta[subj,chains], open('pickles/' + fit + 'theta_s' + str(subj) + '.pkl', 'wb'))
  pickle.dump(phiMu[chains], open('pickles/' + fit + 'phiMus.pkl', 'wb'))
  pickle.dump(phiSigma[chains], open('pickles/' + fit + 'phiSigmas.pkl', 'wb'))
  pickle.dump(weight[:,chains], open('pickles/' + fit + 'weights.pkl', 'wb'))
  pickle.dump(hyperWeight[chains], open('pickles/' + fit + 'hyperWeight.pkl', 'wb'))

def cutChains(fit, dataset):
  p = getParams(fit)
  data = returnData(dataset)

  theta0 = getSubj(0, fit)
  nChains, nmc, nParams = theta0.shape
  nSubj = len(np.unique(data['subj']))
  theta = np.zeros((nSubj,nChains,nmc,nParams))
  theta[0] = theta0
  for subj in xrange(1, nSubj): theta[subj] = getSubj(subj, fit)
  print 'Subjects collected!'  

  phiMu = getPhiMu(fit)
  phiSigma = getPhiSigma(fit)
  weight = getWeight(fit)
  hyperWeight = getHyperWeight(fit)

  chains = range(nChains)
  chains = chains[0:len(p) * 3]
    
  fit += '/'
  for subj in xrange(theta.shape[0]): pickle.dump(theta[subj,chains], open('pickles/' + fit + 'theta_s' + str(subj) + '.pkl', 'wb'))
  pickle.dump(phiMu[chains], open('pickles/' + fit + 'phiMus.pkl', 'wb'))
  pickle.dump(phiSigma[chains], open('pickles/' + fit + 'phiSigmas.pkl', 'wb'))
  pickle.dump(weight[:,chains], open('pickles/' + fit + 'weights.pkl', 'wb'))
  pickle.dump(hyperWeight[chains], open('pickles/' + fit + 'hyperWeight.pkl', 'wb'))
     
def plotChains(subj, fit, dataset):
  data = returnData(dataset)
  data = data[data['subj'] == subj]
  RTs = data['RT']
  theta = getSubj(subj, fit)
  log_dens_like = getLogDensLike(fit)
  p = getParams(fit)
  nChains = theta.shape[0]
  hist(RTs, bins = 30, histtype = 'step', color = 'black', normed = True)
  weight = getWeight(fit)
  for chain in xrange(nChains):
    idx = np.where(weight[subj,chain] == np.max(weight[subj,chain]))[0][0]
    wT = np.max(weight[subj,chain])
    #prms = np.mean(theta[chain], axis = 0) #avg over samples
    prms = theta[chain,idx]
    modelRTs = log_dens_like(prms, data, p, simulate = True, x = 40)[0]
    like = log_dens_like(prms, data, p)
    modelRTs = modelRTs[modelRTs < 3.0]
    color = (.2, chain / float(nChains), .5, .5)
    #hist(modelRTs, bins = 30, histtype = 'step', normed = True)
    plotPosterior(modelRTs, color = color, linewidth = .75)
  xlim(0,)
    
def allSubj(param, fit = '', dataset = ''):
  if 'Annis' in dataset: excludeSubj = True
  else: excludeSubj = False
  data = returnData(dataset, exclude = True, excludeSubj = excludeSubj)
  nSubj = len(np.unique(data['subj']))
  params = getParams(fit)
  for s, subj in enumerate(xrange(nSubj)):
    color = (subj / float(nSubj), subj / float(nSubj), 0, 1)
    theta = getSubj(subj, fit)
    #violinPlot(0, theta[:,:,params.index(param)], color = color)
    #plot(0, np.mean(theta[:,:,params.index(param)]), '.', color = color)
    plotPosterior(theta[:,:,params.index(param)], color = 'black', linewidth = .5, alpha = .6)
  phiMu = getPhiMu(fit)
  plotPosterior(phiMu[:,:,params.index(param)], color = 'red', linewidth = 1.5)

#all params, all subjects
def allSubjectParams(fit, dataset, fontsize = 9):
  params = getParams(fit)
  data = returnData(dataset)
  nSubj = np.unique(data['subj'])
  theta0 = getSubj(0, fit)
  nChains, nmc, nParams = theta0.shape
  nSubj = len(np.unique(data['subj']))
  theta = np.zeros((nSubj,nChains,nmc,nParams))
  theta[0] = theta0
  for subj in xrange(1, nSubj): theta[subj] = getSubj(subj, fit)
  print 'Subjects collected!'

  figure(figsize=(9, 8))
  if len(params) > 42: x, y = 7, 7
  elif len(params) > 36: x, y = 7, 6
  elif len(params) > 30: x, y = 6, 6
  elif len(params) > 25: x, y = 6, 5
  elif len(params) > 20: x, y = 5, 5
  elif len(params) > 16: x, y = 5, 4
  elif len(params) > 12: x, y = 4, 4
  elif len(params) > 9: x, y = 4, 3
  else: x, y = 3, 3

  for p, param in enumerate(params):
    subplot(x, y, p + 1)
    title(param, fontsize = fontsize)
    xticks([], [], fontsize = fontsize)
    yticks(fontsize = fontsize)
    for subj in xrange(nSubj):
      color = (0, subj / float(nSubj), 0, 1)
      violinPlot(0, theta[subj,:,:,params.index(param)], color = color)

def compareRecovery(fit):
  params = getParams(fit)
  phiMu = getPhiMu(fit)
  phiMu_recovery = getPhiMu(fit + '_recovery')
  
  figure(figsize=(9, 8))
  if len(params) > 42: x, y = 7, 7
  elif len(params) > 36: x, y = 7, 6
  elif len(params) > 30: x, y = 6, 6
  elif len(params) > 25: x, y = 6, 5
  elif len(params) > 20: x, y = 5, 5
  elif len(params) > 16: x, y = 5, 4
  elif len(params) > 12: x, y = 4, 4
  elif len(params) > 9: x, y = 4, 3
  else: x, y = 3, 3

  for p, param in enumerate(params):
    subplot(x, y, p + 1)
    title(param, fontsize = fontsize)
    xticks([], [], fontsize = fontsize)
    yticks(fontsize = fontsize)
    violinPlot(0, phiMu[:,:,params.index(param)], color = 'black')
    violinPlot(0, phiMu_recovery[:,:,params.index(param)], color = 'red')

def allSubjectCorrelation(fit, dataset, fontsize = 9):
  params = getParams(fit)
  data = returnData(dataset)
  nSubj = np.unique(data['subj'])
  theta0 = getSubj(0, fit)
  nChains, nmc, nParams = theta0.shape
  nSubj = len(np.unique(data['subj']))
  theta = np.zeros((nSubj,nChains,nmc,nParams))
  theta[0] = theta0
  for subj in xrange(1, nSubj): theta[subj] = getSubj(subj, fit)
  print 'Subjects collected!'

  figure(figsize=(9, 8))
  x, y = len(params) - 1, len(params) - 1

  for i, param in enumerate(params):
    params2 = params[i:]
    for j, param in enumerate(params2):
      p1 = theta[:,:,i].flatten()
      p2 = theta[:,:,j].flatten()
      corr = pearsonr(p1, p2)[0]
      subplot(x,y,(y * i) + j+2)
      scatter(x, y, alpha = .035, s = 6.5, color = 'black')
      title(param + ' ,: r = ' + str(round(corr, 3)), fontsize = fontsize)
    
def compareRecovery(fit, subj = 0):
  params = getParams(fit)
  theta = getSubj(subj, fit)
  theta_recovery = getSubj(subj, fit + '_recovery')  

  figure(figsize=(9, 8))
  if len(params) > 42: x, y = 7, 7
  elif len(params) > 36: x, y = 7, 6
  elif len(params) > 30: x, y = 6, 6
  elif len(params) > 25: x, y = 6, 5
  elif len(params) > 20: x, y = 5, 5
  elif len(params) > 16: x, y = 5, 4
  elif len(params) > 12: x, y = 4, 4
  elif len(params) > 9: x, y = 4, 3
  else: x, y = 3, 3
  
  for p, param in enumerate(params):
    subplot(x, y, p + 1)
    title(param, fontsize = fontsize)
    xticks([], [], fontsize = fontsize)
    yticks(fontsize = fontsize)
    axhline(np.median(theta[:,:,params.index(param)]), color = 'black', linestyle = '--')
    violinPlot(0, theta_recovery[:,:,params.index(param)], color = 'red')

def recoveryAllSubjects(fit, dataset, param):
  
  data = returnData(dataset)
  nSubj = len(np.unique(data['subj']))
  params = getParams(fit)

  figure(figsize=(9, 8))
  if nSubj > 42: x, y = 7, 7
  elif nSubj > 36: x, y = 7, 6
  elif nSubj > 30: x, y = 6, 6
  elif nSubj > 25: x, y = 6, 5
  elif nSubj > 20: x, y = 5, 5
  elif nSubj > 16: x, y = 5, 4
  elif nSubj > 12: x, y = 4, 4
  elif nSubj > 9: x, y = 4, 3
  else: x, y = 3, 3

  p = 0
  for subj in xrange(nSubj):
    theta = getSubj(subj, fit)
    theta_recovery = getSubj(subj, fit + '_recovery')  

    subplot(x, y, subj + 1)
    title(param, fontsize = fontsize)
    xticks([], [], fontsize = fontsize)
    yticks(fontsize = fontsize)
    gen = np.median(theta[:,:,params.index(param)])
    recov = theta_recovery[:,:,params.index(param)]
    axhline(gen, color = 'black', linestyle = '--')
    violinPlot(1, recov, color = 'red')
    p += np.mean(recov > gen)
  p = p / float(nSubj)
  suptitle('p = ' + str(p), fontsize = 14)

##### PLOT POSTERIOR #####

def plotPosterior(arr, nBins = 250, color = '', linestyle = '-', linewidth = 1, alpha = 1.0, returnGraph = False, thin = 1, **kwargs):
  vals = arr.reshape(arr.size).astype(np.double)
  vals = vals[np.isfinite(vals)]
  vals = vals[arange(0, len(vals), thin)] #thinning
  kde = KDE(vals)
  kde.fit(gridsize = nBins)
  if color != '': a = plot(kde.support, kde.density, color = color, linestyle = linestyle, linewidth = linewidth, alpha = alpha)[0]
  else: return plot(kde.support, kde.density, linestyle = linestyle, linewidth = linewidth, alpha = alpha)[0]
  if returnGraph == True: return a

##### TRACE PLOTS #####

#args: array containing parameters, parameter index, parameter list
def tracePlot(arr, param, params):
  p = arr[:,:,params.index(param)]
  nChains, nSamples = p.shape[0], p.shape[1]
  p[np.isnan(p)] = -1
  for chain in xrange(nChains): plot(arange(nSamples), p[chain], alpha = .25)

#all trace plots for either a subject (theta) or group level (phiMu or phiSigma)
def allTracePlots(arr, params, fontsize = 9, conv = True, newFigure = True):
  if conv == True: gr = gelman_rubin(arr)
  if newFigure == True: figure(figsize=(9, 8))
  if len(params) > 64: x, y = 9, 8
  elif len(params) > 56: x, y = 8, 8
  elif len(params) > 49: x, y = 8, 7
  elif len(params) > 42: x, y = 7, 7
  elif len(params) > 36: x, y = 7, 6
  elif len(params) > 30: x, y = 6, 6
  elif len(params) > 25: x, y = 6, 5
  elif len(params) > 20: x, y = 5, 5
  elif len(params) > 16: x, y = 5, 4
  elif len(params) > 12: x, y = 4, 4
  elif len(params) > 9: x, y = 4, 3
  else: x, y = 3, 3
  for p, param in enumerate(params):
    subplot(x, y, p + 1)
    if p == 0: graphs = tracePlot(arr, param, params)
    else: tracePlot(arr, param, params)
    if conv == True:
      stat = np.around(gr[p], 3)
      title(param + ': ' + str(stat), fontsize = fontsize)
    else: title(param)
    xticks(fontsize = fontsize)
    yticks(fontsize = fontsize)
  subplots_adjust(left = .06, bottom = .05, right = .94, top = .94, hspace = .34, wspace = .20)

def allPosteriors(arr, params, fontsize = 9, alpha = .5, color = 'black'):
  for p, param in enumerate(params):
    plotPosterior(arr[:,:,params.index(param)].flatten(), color = color, alpha = alpha)

def allSubjects():
  figure(figsize = (10,10))
  data = returnData('srwp')
  nSubj = len(np.unique(data['subj']))
  p = getParams('srwpDDM_lrdc')
  for subj in xrange(nSubj):
    subplot(6,7,subj + 1)
    theta = getSubj(subj, 'srwpDDM_lrdc')
    allPosteriors(theta, p, alpha = .5, color = 'green')
    theta = getSubj(subj, 'srwpDDM_lrdc2')
    allPosteriors(theta, p, color = 'red', alpha = .5)
    title(subj)
    xlim(0, 1.0)
  subplots_adjust(left = .06, bottom = .05, right = .94, top = .94, hspace = .34, wspace = .20)

def histMean(arr, params, fontsize = 9, burnin = 500):
  figure(figsize=(9, 8))
  if len(params) > 42: x, y = 7, 7
  elif len(params) > 36: x, y = 7, 6
  elif len(params) > 30: x, y = 6, 6
  elif len(params) > 25: x, y = 6, 5
  elif len(params) > 20: x, y = 5, 5
  elif len(params) > 16: x, y = 5, 4
  else: x, y = 4, 4
  for p, param in enumerate(params):
    subplot(x, y, p + 1)
    pr = arr[:,burnin:,params.index(param)]
    hist(pr.reshape(pr.size), bins = 300, histtype = 'step')
    title(param, fontsize = fontsize)
    axvline(np.mean(pr), color = 'black')
    axvline(np.median(pr), color = 'red')
    xticks(fontsize = fontsize)
    yticks(fontsize = fontsize)
  subplots_adjust(left = .06, bottom = .05, right = .94, top = .94, hspace = .34, wspace = .20)

#use this to plot either weight or like for a given subject
#...to plot for all subjects, you can collapse weight across the subject dimension using np.sum(weight, axis = 0)
#...the "axis" command specifies the dimension over which to sum, and the "0" specifies the subject dimension
def plotWeight(weight):
  nChains, nSamples = weight.shape
  for chain in xrange(nChains): plot(arange(nSamples), weight[chain], alpha = .22)

#### PR SUBMISSION ####

#compare parameters for the first block vs. all

def compareParamsOI(fontsize = 9):
  params = ['logVti', 'logVsu', 'logVtt', 'UssDelay', 'gamma', 'aShortDelay', 'aLongDelay', 'aShortImmed', 'aLongImmed', 'aSlopeImmed', 'aSlopeDelay']
  titles = ['Vti', 'Vsu', 'Vtt', 'Uss', 'gamma', 'aShortDelay', 'aLongDelay', 'aShortImmed', 'aLongImmed', 'aSlopeImmed', 'aSlopeDelay']
  colors = ['blue', 'red']
  graphs = []
  for m, model in enumerate(['oiCOCODDMlb_Uss', 'oi0COCODDMlb_Uss']):
    phiMu = getPhiMu(model)
    p = getParams(model)
    for i, param in enumerate(params):
      subplot(3, 4, i + 1)
      prm = phiMu[:,:,p.index(param)]
      graph = plotPosterior(prm, color = colors[m], returnGraph = True)
      title(titles[i], fontsize = fontsize * 1.2)
      xticks(fontsize = fontsize)
      yticks(fontsize = fontsize)
      
      if i == 0: graphs.append(graph)
      if (m == 1) & (i == 0):
        legend(graphs, ['Within', 'Between'], prop = legendFont, loc = 'upper left')
    
def compareVsuOI(fontsize = 9):
  data = returnData('oi0')
  theta0 = getSubj(0, 'oi0COCODDMlb_Uss')
  p = getParams('oi0COCODDMlb_Uss')
  subjectList = np.unique(data['subj'])
  nSubj = len(subjectList)
  theta = np.zeros((nSubj,theta0.shape[0],theta0.shape[1],theta0.shape[2]))
  theta[0] = theta0
  for i in xrange(1, nSubj):
    theta[i] = getSubj(i, 'oi0COCODDMlb_Uss')
  
  LongDelay = []
  ShortDelay = []
  ShortImmed = []
  LongImmed = []
  short_idx, long_idx = data['length'] == 'Short', data['length'] == 'Long'
  immed_idx, delay_idx = data['delay'] == 'Immed', data['delay'] == 'Delay'

  #get subject labels for each condition
  for i in xrange(nSubj):
    subj_idx = data['subj'] == subjectList[i]
    if len(data[subj_idx & short_idx & immed_idx]['response']) > 0: ShortImmed.append(i)
    elif len(data[subj_idx & long_idx & immed_idx]['response']) > 0: LongImmed.append(i)
    elif len(data[subj_idx & short_idx & delay_idx]['response']) > 0: ShortDelay.append(i)
    elif len(data[subj_idx & long_idx & delay_idx]['response']) > 0: LongDelay.append(i)

  title('Vsu', fontsize = fontsize * 1.4)
  a = plotPosterior(theta[ShortImmed,:,:,p.index('logVsu')], color = 'blue', returnGraph = True)
  b = plotPosterior(theta[ShortDelay,:,:,p.index('logVsu')], color = 'blue', linestyle = ':', returnGraph = True)
  c = plotPosterior(theta[LongImmed,:,:,p.index('logVsu')], color = 'red', returnGraph = True)
  d = plotPosterior(theta[LongDelay,:,:,p.index('logVsu')], color = 'red', linestyle = ':', returnGraph = True)
  e = plotPosterior(theta[:,:,:,p.index('logVsu')], color = 'black', linewidth = 2, returnGraph = True)
  legend([a,b,c,d,e], ['Short Imm', 'Short Del', 'Long Imm', 'Long Del', 'Overall'], loc = 'lower left', prop = legendFont, ncol = 3)
  
  '''
  subplot(122)
  title('a', fontsize = fontsize * 1.4)
  plotPosterior(theta[ShortImmed,:,:,p.index('aShortImmed')], color = 'blue')
  plotPosterior(theta[ShortDelay,:,:,p.index('aShortDelay')], color = 'blue', linestyle = ':')
  plotPosterior(theta[LongImmed,:,:,p.index('aLongImmed')], color = 'red')
  plotPosterior(theta[LongDelay,:,:,p.index('aLongDelay')], color = 'red', linestyle = ':')'''
  
def plotAnnis(fit = 'AnnisSD', dataset = 'AnnisSD'):
  data = returnData(dataset)
  phiMu = getPhiMu(fit)
  p = getParams(fit)
  if 'DDM' in fit: baseParams = ['t0', 'st0', 'sz', 'a', 'z', 'vlure', 'vtarget', 'svLure', 'svTarget']
  elif 'LBA' in fit: baseParams = ['t0', 'A', 'bOld', 'bNew', 'vlure', 'vtarget', 'svLure', 'svTarget']
  conds = np.unique(data['ISItask'])
  colors = ['blue', 'red', 'green']
  graphs = []
  for i, param in enumerate(baseParams):
    if param + conds[0] in p:
      for c, cond in enumerate(conds):
        if i == 0: graphs.append(violinPlot(i, phiMu[:,:,p.index(param + cond)], color = colors[c], returnGraph = True)[0])
        else: violinPlot(i, phiMu[:,:,p.index(param + cond)], color = colors[c])
    else:
      if i == 0: graphs.append(violinPlot(i, phiMu[:,:,p.index(param)], color = 'black', returnGraph = True))
      else: violinPlot(i, phiMu[:,:,p.index(param)], color = 'black')      
  xticks(range(len(baseParams)), baseParams)
  ylim(0,)
  legend(graphs, conds, prop = legendFont, loc = 'upper left')

def rhat(theta):
  m, n = theta.shape
  W = np.var(theta, axis = 0) #W: variance within each chain
  W = np.mean(W)
  B = np.var(theta, axis = 1) #B: between chains variance
  B = np.mean(B)
  mu_hat = np.mean(theta)
  V_hat = sigma_hat2 + np.mean(B_n)
  d = (2*V_hat**2) / np.var(V_hat)
  return np.sqrt(((d+3) * V_hat) /((d+1)*W))

def acceptanceRatesPhi(arr):
  nChains, nSamples = arr.shape
  return np.array([1.0 - np.mean([arr[chain,i] == arr[chain,i-1] for i in xrange(1,nSamples)]) for chain in xrange(nChains)])

def acf(series):
  n = len(series)
  data = np.asarray(series)
  M = np.mean(data)
  c0 = np.sum((data - M) ** 2) / float(n)

  def r(h):
    acf_lag = ((data[:n - h] - M) * (data[h:] - M)).sum() / float(n) / c0
    return round(acf_lag, 3)
  x = np.arange(n) # Avoiding lag 0 calculation
  acf_coeffs = map(r, x)
  return acf_coeffs

def acfChains(arr):
  global k
  nChains, nSamples = arr.shape
  k = np.mean(np.array([acf(arr[chain]) for chain in xrange(nChains)]), axis = 0)
  plot(np.arange(1,nSamples), k[1:])
  xlabel('Lag')
  ylabel('Correlation')
  xlim(0,200)

def lag1acf(arr):
  nChains, nSamples = arr.shape
  return np.array([acf(arr[chain])[1] for chain in xrange(nChains)])

#test of linear interpolation function
from de import linear_interpolation
from deFuncs import twoNearest
def testLinear():
  phiMu = getPhiMu('oiCOCODDMt')
  p = getParams('oiCOCODDMt')
  UssDelay = phiMu[:,:,p.index('UssDelay')].flatten().astype(np.double)
  kde = KDE(UssDelay)
  kde.fit(gridsize = 4000)
  a, b = np.min(kde.support), np.max(kde.support)
  vals = np.random.uniform(a, b, size = 20)
 
  new = linear_interpolation(vals, kde.support, kde.density)
  for i in xrange(len(vals)):
    subplot(5, 4, i + 1)
    val = vals[i]
    args = (np.abs(kde.support - val)).argsort()
    x0, x1 = kde.support[args[0]], kde.support[args[1]]
    y0, y1 = kde.density[args[0]], kde.density[args[1]]
    plot([x0,x1], [y0,y1], color = 'gray')
    plot(val, new[i], '*')

def plotalphabeta(m = .5, s = 2, plot = 's'):
  global x, alpha, beta
  if plot == 's':
    s = np.arange(.01, 2, .01)
    x = s
  elif plot == 'm':
    m = np.arange(.01, 1, .01)
    x = m
  subplot(121)
  alpha = -1 * (m * (s**2 + m**2 - m)) / s**2
  plot(x, alpha)
  title('Alpha')
  subplot(122)
  beta = ((s**2 + m**2 - m) * (m - 1)) / s**2
  plot(x, beta)
  title('Beta')

def getcsim(subj = 0):
  data = returnData('Criss2BEAGLE')
  subjectList = np.unique(data['subj'])
  data = data[data['subj'] == subjectList[subj]]
  
  power = 2.0
  scale = 1.0
  
  
  pairs = data['contextCos'].reshape(len(data),50)
  c_ss = scale * np.sum(np.sign(pairs) * np.exp(power * np.log(np.abs(pairs))), axis = -1)
  c_ss /= 50.
  
  return c_ss

  
  
