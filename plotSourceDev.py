from extract import *
from plotFit import violinPlot
from pylab import *
from scipy.stats import linregress
import os
from copy import deepcopy

legendFont = matplotlib.font_manager.FontProperties(size=7)

### PLOT POSTERIOR PREDICTIVES ###

def plotRates(fit = 'sourceDev4COCOfreq', color = 'blue', dataColor = 'black', fontsize = 9, nRows = 1, row = 0, titles = True, showLegend = False):

  legendFont = matplotlib.font_manager.FontProperties(size=8)

  rates, modelRates = pickle.load(open('pickles/' + fit + '/avgPred.pkl', 'rb'))
  rates, modelRates = np.nanmean(rates, axis = 0), np.nanmean(modelRates, axis = 0)
  scale = 1/135.
  
  lw_data = .8
  ls = ':'
  model_lw = .7
  nCol = 2

  dataset = dataName(fit)
  data = returnData(dataset)

  rates, errors = getCondMeans(data) #calculate means and standard errors for everything

  ## ITEM PLOT ##

  subplot(nRows,2,(row * nCol) + 1)
  if titles == True: title('Item Recognition Rates', fontsize = fontsize * 1.2)
  a = errorbar(np.arange(3) + 1, rates[5,:,0], yerr = errors[5,:,0], marker = '.', color = dataColor, linewidth = lw_data, linestyle = ls)[0]
  errorbar(np.arange(3) + 1, rates[4,:,0], yerr = errors[4,:,0], marker = '.', color = dataColor, linewidth = lw_data, linestyle = ls)
  errorbar(np.arange(3) + 5, rates[5,:,1], yerr = errors[5,:,1], marker = '.', color = dataColor, linewidth = lw_data, linestyle = ls)
  errorbar(np.arange(3) + 5, rates[4,:,1], yerr = errors[4,:,1], marker = '.', color = dataColor, linewidth = lw_data, linestyle = ls)
  for i in [4,5]:
    for j in xrange(3):
      for k in xrange(2):
        if k == 0: x = j+1
        else: x = j+5
        violinPlot(x, modelRates[i,j,k], alpha = .8, linewidth = model_lw, scale = scale, color = color)
  b = errorbar(2, rates[6,1,0], yerr = errors[6,1,0], marker = '*', color = dataColor)[0]
  errorbar(6, rates[6,1,1], yerr = errors[6,1,1], marker = '*', color = dataColor)
  c = violinPlot(2, modelRates[6,1,0], alpha = .8, linewidth = model_lw, scale = scale, color = color, returnGraph = True)[0]
  violinPlot(6, modelRates[6,1,1], alpha = .8, linewidth = model_lw, scale = scale, color = color)

  xticks([1,2,3,5,6,7], ['PW', 'M\nHF', 'Long', 'PW', 'M\nLF', 'Long'], fontsize = fontsize)
  yticks(fontsize = fontsize)
  ylabel('$p(yes)$', fontsize = fontsize)
  xlim(0, 8)
  ylim(0, 1.01)

  ## SOURCE PLOT ##

  subplot(nRows,2,(row * nCol) + 2)
  if titles == True: title('Source Memory', fontsize = fontsize * 1.2)
  errorbar(np.arange(3) + 1, np.mean(rates[0:2,:,0], axis = 0), yerr = np.mean(errors[0:2,:,0]), marker = '.', color = dataColor, linewidth = lw_data, linestyle = ls)
  errorbar(np.arange(3) + 5, np.mean(rates[0:2,:,1], axis = 0), yerr = np.mean(errors[0:2,:,1]), marker = '.', color = dataColor, linewidth = lw_data, linestyle = ls)
  for j in xrange(3):
    for k in xrange(2):
      if k == 0: x = j+1
      else: x = j+5
      r = np.mean(modelRates[0:2,j,k], axis = 0)
      violinPlot(x, r, alpha = .8, linewidth = model_lw, scale = scale, color = color)
  errorbar(2, np.mean(rates[2:4,1,0], axis = 0), yerr = np.mean(errors[2:4,1,0]), marker = '*', color = dataColor)
  errorbar(6, np.mean(rates[2:4,1,1], axis = 0), yerr = np.mean(errors[2:4,1,1]), marker = '*', color = dataColor)
  violinPlot(2, np.mean(modelRates[2:4,1,0], axis = 0), alpha = .8, linewidth = model_lw, scale = scale, color = color)
  violinPlot(6, np.mean(modelRates[2:4,1,1], axis = 0), alpha = .8, linewidth = model_lw, scale = scale, color = color)

  xticks([1,2,3,5,6,7], ['PW', 'M\nHF', 'Long', 'PW', 'M\nLF', 'Long'], fontsize = fontsize)
  yticks(fontsize = 9) 
  ylabel('$p(correct)$', fontsize = fontsize)
  xlim(0, 8)
  ylim(0, 1.01)
  axhline(.5, linestyle = '--', color = 'black')

  if showLegend == True: legend([a,b,c], ['Data (weak)', 'Data (strong)', 'Model'], loc = 'lower right', prop = legendFont)

def plotRatesMarginal(fit = 'sourceDev4COCOfreq', color = 'blue', dataColor = 'black', fontsize = 8, alpha = .65, nRows = 1, row = 0, msize = 6.75, titles = True, showLegend = False):

  global modelRates
  legendFont = matplotlib.font_manager.FontProperties(size=6.5)

  rates, modelRates = pickle.load(open('pickles/' + fit + '/avgPred.pkl', 'rb'))
  rates, modelRates = np.nanmean(rates, axis = 0), np.nanmean(modelRates, axis = 0)
  scale = 1/150.
  
  lw_data = .8
  ls = ':'
  model_lw = .85
  nCol = 4

  dataset = dataName(fit)
  data = returnData(dataset)

  ratesL, errorsL, ratesW, errorsW = getCondMeansMarginal(data) #calculate means and standard errors for everything

  ### ITEM PLOTS ###

  #LENGTH/STRENGTH
  subplot(nRows,4,(row * nCol) + 1)
  if titles == True: title('Item\nLL/LS', fontsize = fontsize * 1.2)
  a = errorbar(np.arange(3) + 1, ratesL[5], yerr = errorsL[5], marker = '.', markersize = msize, color = dataColor, linewidth = lw_data, linestyle = ls)[0]
  errorbar(np.arange(3) + 1, ratesL[4], yerr = errorsL[4], marker = '.', markersize = msize, color = dataColor, linewidth = lw_data, linestyle = ls)
  for i in [4,5]:
    for j in xrange(3):
      x = j+1
      violinPlot(x, np.nanmean(modelRates[i,j], -2), alpha = alpha, linewidth = model_lw, scale = scale, color = color) #average over word frequency here (last dimension is samples, second to last is WF)

  #strong items
  b = errorbar(2, ratesL[6,1], yerr = errorsL[6,1], marker = '*', markersize = msize, color = dataColor)[0]
  c = violinPlot(2, np.nanmean(modelRates[6,1], -2), alpha = alpha, linewidth = model_lw, scale = scale, color = color, returnGraph = True)[0]

  xticks([1,2,3], ['PW', 'M', 'Long'], fontsize = fontsize)
  yticks(fontsize = fontsize)
  ylabel('$p(yes)$', fontsize = fontsize)
  xlim(.5, 3.5)
  ylim(0, 1.01)

  #WORD FREQUENCY
  subplot(nRows,4,(row * nCol) + 2)
  if titles == True: title('Item\nWF', fontsize = fontsize * 1.2)
  a = errorbar(np.arange(2) + 1, ratesW[5], yerr = errorsW[5], marker = '.', markersize = msize, color = dataColor, linewidth = lw_data, linestyle = ls)[0]
  errorbar(np.arange(2) + 1, ratesW[4], yerr = errorsW[4], marker = '.', markersize = msize, color = dataColor, linewidth = lw_data, linestyle = ls)
  for i in [4,5]:
    for j in xrange(2):
      x = j+1
      violinPlot(x, np.nanmean(modelRates[i,:,j], axis = 0), alpha = alpha, linewidth = model_lw, scale = scale, color = color) #average over length/strength here (last dimension is samples, second to last is LL/LS)

  #strong items
  b = errorbar(np.arange(2) + 1, ratesW[6], yerr = errorsW[6], marker = '*', markersize = msize, color = dataColor)[0]
  for i in xrange(2):
    c = violinPlot(i+1, modelRates[6,1,i], alpha = alpha, linewidth = model_lw, scale = scale, color = color, returnGraph = True)[0]

  xticks([1,2], ['HF', 'LF'], fontsize = fontsize)
  yticks(fontsize = fontsize)
  xlim(.5, 2.5)
  ylim(0, 1.01)
  yticks([])


  ### SOURCE PLOTS ###

  #LENGTH/STRENGTH
  subplot(nRows,4,(row * nCol) + 3)
  if titles == True: title('Source\nLL/LS', fontsize = fontsize * 1.2)
  errorbar(np.arange(3) + 1, np.mean(ratesL[0:2], axis = 0), yerr = np.mean(errorsL[0:2]), marker = '.', markersize = msize, color = dataColor, linewidth = lw_data, linestyle = ls)
  for j in xrange(3):
    x = j+1
    r = np.mean(modelRates[0:2,j], axis = (0,-2))
    violinPlot(x, r, alpha = alpha, linewidth = model_lw, scale = scale, color = color)
  #strong items
  errorbar(2, np.mean(ratesL[2:4,1], axis = 0), yerr = np.mean(errorsL[2:4,1]), marker = '*', markersize = msize, color = dataColor)
  violinPlot(2, np.mean(modelRates[2:4,1], axis = (0,-2)), alpha = alpha, linewidth = model_lw, scale = scale, color = color)
  xticks([1,2,3], ['PW', 'M', 'Long'], fontsize = fontsize)
  yticks(fontsize = 9) 
  ylabel('$p(correct)$', fontsize = fontsize)
  xlim(.5, 3.5)
  ylim(0, 1.01)
  yticks(fontsize = fontsize)
  axhline(.5, linestyle = '--', color = 'black')

  if showLegend == True: legend([a,b,c], ['Data (weak)', 'Data (strong)', 'Model'], loc = 'lower right', prop = legendFont)

  #WORD FREQUENCY
  subplot(nRows,4,(row * nCol) + 4)
  if titles == True: title('Source\nWF', fontsize = fontsize * 1.2)
  errorbar(np.arange(2) + 1, np.mean(ratesW[0:2], axis = 0), yerr = np.mean(errorsW[0:2]), marker = '.', markersize = msize, color = dataColor, linewidth = lw_data, linestyle = ls)
  for j in xrange(2):
    x = j+1
    r = np.mean(modelRates[0:2,:,j], axis = (0,1))
    violinPlot(x, r, alpha = alpha, linewidth = model_lw, scale = scale, color = color)
  #strong items
  errorbar(np.arange(2) + 1, np.mean(ratesW[2:4], axis = 0), yerr = np.mean(errorsW[2:4,1]), marker = '*', markersize = msize, color = dataColor)
  for j in xrange(2):
    violinPlot(j+1, np.mean(modelRates[2:4,1,j], axis = 0), alpha = alpha, linewidth = model_lw, scale = scale, color = color)
  xticks([1,2], ['HF', 'LF'], fontsize = fontsize)
  yticks(fontsize = 9) 
  xlim(.5, 2.5)
  ylim(0, 1.01)
  yticks([])
  axhline(.5, linestyle = '--', color = 'black')

def plotRatesAll(fontsize = 9, version = '', color = 'blue', newFigure = True):

  if newFigure == True:
    figure(figsize=(7.25,6.25))
    subplots_adjust(left = .1, bottom = .08, right = .9, top = .94, hspace = .37, wspace = .24)

  fits = ['sourceDev4COCOfreq' + version, 'sourceDev8COCOfreq' + version, 'sourceDevAdultsCOCOfreq' + version]
  labels = ['Age 4', 'Age 8', 'Adults']

  for f, fit in enumerate(fits):
    if f == 0: showLegend, titles = True, True
    else: showLegend, titles = False, False
    plotRates(fit, nRows = 3, row = f, titles = titles, showLegend = showLegend, fontsize = fontsize)

    subplot(3,2,(f * 2) + 1)
    text(-2.4, .5, labels[f], ha = 'center', va = 'center', fontsize = fontsize * 1.1, weight = 'bold', rotation = 'vertical')

def plotRatesAllMarginal(version = '', fontsize = 7.5, color = 'blue', newFigure = True):

  if newFigure == True:
    figure(figsize=(7.5,5.8))
    subplots_adjust(left = .1, bottom = .08, right = .9, top = .94, hspace = .3, wspace = .65)

  fits = ['sourceDev45COCOfreq' + version, 'sourceDev78COCOfreq' + version, 'sourceDevAdultsCOCOfreq' + version]
  labels = ['Age 4-5', 'Age 7-8', 'Adults']

  for f, fit in enumerate(fits):
    if f == 0: showLegend, titles = True, True
    else: showLegend, titles = False, False
    plotRatesMarginal(fit, nRows = 3, row = f, titles = titles, showLegend = showLegend, fontsize = fontsize, color = color)

    subplot(3,4,(f * 4) + 1)
    text(-1.4, .5, labels[f], ha = 'center', va = 'center', fontsize = fontsize * 1.1, weight = 'bold', rotation = 'vertical')

from plotFit import plotPosterior
def plotParams():
  phiMu = getPhiMu('sourceDev4COCO')
  phiMu2 = getPhiMu('sourceDev8COCO')
  p = getParams('sourceDev4COCO')
  
  for i, param in enumerate(p):
    subplot(3,4,i+1)
    plotPosterior(phiMu[:,:,p.index(param)], color = 'blue')
    plotPosterior(phiMu2[:,:,p.index(param)], color = 'red')
    title(param, fontsize = 9)
    xticks(fontsize = 8)
    yticks([], [])

def compareParams(version = ''):
  params = ['logVti', 'crit', 'critSource', 'logrWeak', 'logrStrong']

  phiMu = getPhiMu('sourceDev4COCOfreq' + version)
  p = getParams('sourceDev4COCOfreq' + version)
  phiMu2 = getPhiMu('sourceDev4COCOfreq' + version)
  p2 = getParams('sourceDev4COCOfreq' + version)

  if 'rWeak' in p: params.append('rWeak')
  elif 'logrWeak' in p: params.append('logrWeak')

  for i, param in enumerate(params):
    subplot(3,4,i+1)
    plotPosterior(phiMu[:,:,p.index(param)], color = 'blue')
    plotPosterior(phiMu2[:,:,p2.index(param)], color = 'red')
    title(param)

def compareAgesAll(fontsize = 8, version = ''):
  fits = ['sourceDev45COCOfreq' + version, 'sourceDev78COCOfreq' + version, 'sourceDevAdultsCOCOfreq' + version]
  colors = ['blue', 'purple', 'red']
  p = getParams(fits[0])
  labels = ['4-5yrs', '7-8yrs', 'Adults']

  graphs = []
  for f, fit in enumerate(fits):
    phiMu = getPhiMu(fit)
    p = getParams(fit)
    for i, param in enumerate(p):
      subplot(3,3,i+1)
      vals = phiMu[:,:,p.index(param)]
      if param == 'logrStrong': vals += 1
      if i == 0: graphs.append(plotPosterior(vals, returnGraph = True, color = colors[f]))
      else: plotPosterior(vals, color = colors[f])
      label = getLabel(param)
      title(label, fontsize = fontsize * 1.5)
      yticks([], [])
      xticks(fontsize = fontsize)
      
      if f == 2: legend(graphs, labels, loc = 'upper left', prop = legendFont)
  tight_layout()

   
  
def getLabel(param):
  if param == 'logVti': label = r'$log(\sigma_{ti}^2)$'
  elif param == 'logVsu': label = r'$log(\sigma_{su}^2)$'
  elif param == 'crit': label = r'$\phi_{item}$'
  elif param == 'critSource': label = r'$\phi_{source}$'
  elif param == 'logVaa': label = r'$log(\sigma_{aa}^2)$'
  elif param == 'logVab': label = r'$log(\sigma_{ab}^2)$'
  elif param == 'logVac': label = r'$log(\sigma_{ac}^2)$'
  elif param == 'logrStrong': label = r'$log(r_{strong})$'
  elif param == 'logrWeak': label = r'$log(r_{weak})$'
  elif param == 'rWeak': label = r'$r_{weak}$'
  else: label = param
  return label

from scipy.stats import sem
def plotHalves(nBlocks = 2):
  datasets = ['sourceDev4', 'sourceDev8', 'sourceDevAdults']
  groups = ['Age 4', 'Age 8', 'Adults']
  colors = ['green', 'blue', 'red']
  markers = ['o', '.', '*']
  offsets = [-.05, 0, .05]
  for d, dataset in enumerate(datasets):
    data = returnData(dataset)
    subjectList = np.unique(data['subj'])
    subplot(1,3,d+1)
    title(groups[d]) 
    for c, cond in enumerate(['pure', 'mixed', 'long']):
      if cond == 'mixed': types = ['weak', 'lure', 'strong']
      else: types = ['weak', 'lure']
      
      if cond == 'long': nTrials = 20
      else: nTrials = 10

      if cond in ['pure', 'mixed']:
        if nBlocks == 2: n = 1
        elif nBlocks == 4: n = 2
      else: n = nBlocks
      block = nTrials / nBlocks

      print cond, n
      for o, oldnew in enumerate(types):

        means = np.zeros(n)
        errors = np.zeros(n)
        for h in xrange(n):
          t_idx = (data['trial'] >= (block * h)) & (data['trial'] < ((h+1) * block))
          rate = [np.nanmean(data[(data['type'] == oldnew) & (data['cond'] == cond) & (data['subj'] == subject) & t_idx]['response'] < 2) for subject in subjectList]
          means[h] = np.nanmean(rate)
          errors[h] = sem(rate, nan_policy = 'omit')
        errorbar(np.arange(n) + offsets[c], means, yerr = errors, marker = markers[o], color = colors[c], ecolor = 'black')
    if d > 0: yticks([], [])
    else: ylabel('$p(yes)$')
    xticks(np.arange(nBlocks), np.arange(nBlocks) + 1)
    xlabel('Test Blocks')
    ylim(0, 1)

### DATA PROCESSING ###

def getCondMeans(data):

  subjectList = np.unique(data['subj'])
  nSubj = len(subjectList)

  counts = np.zeros((nSubj,5,3,2,3)) #type (lure, sourceAweak, sourceBweak, sourceAstrong, sourceBstrong) x cond x wf
  for w, wf in enumerate(['HF', 'LF']):
    wf_idx = data['wf'] == wf
    for c, cond in enumerate(['pure', 'mixed', 'long']):
      cond_idx = data['cond'] == cond
      for i in xrange(3):
        counts[:,4,c,w,i] = [np.sum(data[wf_idx & cond_idx & (data['type'] == 'lure') & (data['subj'] == subject)]['response'] == i) for subject in subjectList]
        counts[:,0,c,w,i] = [np.sum(data[wf_idx & cond_idx & (data['type'] == 'weak') & (data['source'] == 0) & (data['subj'] == subject)]['response'] == i) for subject in subjectList]
        counts[:,1,c,w,i] = [np.sum(data[wf_idx & cond_idx & (data['type'] == 'weak') & (data['source'] == 1) & (data['subj'] == subject)]['response'] == i) for subject in subjectList]
        if cond == 'mixed':
          counts[:,2,c,w,i] = [np.sum(data[wf_idx & cond_idx & (data['type'] == 'strong') & (data['source'] == 0) & (data['subj'] == subject)]['response'] == i) for subject in subjectList]
          counts[:,3,c,w,i] = [np.sum(data[wf_idx & cond_idx & (data['type'] == 'strong') & (data['source'] == 1) & (data['subj'] == subject)]['response'] == i) for subject in subjectList]

  dataRates = np.zeros((nSubj,7,3,2))

  sm = np.sum(counts[:,0,:,:,0:2], axis = -1).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,0] = counts[:,0,:,:,0] / sm

  sm = np.sum(counts[:,1,:,:,0:2], axis = -1).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,1] = counts[:,1,:,:,1] / sm

  sm = np.sum(counts[:,2,:,:,0:2], axis = -1).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,2] = counts[:,2,:,:,0] / sm

  sm = np.sum(counts[:,3,:,:,0:2], axis = -1).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,3] = counts[:,3,:,:,1] / sm

  sm = np.sum(counts[:,4], axis = -1).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,4] = (counts[:,4,:,:,0] + counts[:,4,:,:,1]) / sm #FAR = (source A + source B) / CR

  sm = np.sum(counts[:,0:2], axis = (1,-1)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,5] = np.sum(counts[:,0:2,:,:,[0,1]], axis = (1,-1)) / sm

  sm = np.sum(counts[:,2:4], axis = (1,-1)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,6] = np.sum(counts[:,2:4,:,:,[0,1]], axis = (1,-1)) / sm
 
  rates = np.mean(dataRates, axis = 0)
  errors = sem(dataRates, axis = 0)
    
  return rates, errors

def getCondMeansMarginal(data):

  subjectList = np.unique(data['subj'])
  nSubj = len(subjectList)

  counts = np.zeros((nSubj,5,3,2,3)) #type (lure, sourceAweak, sourceBweak, sourceAstrong, sourceBstrong) x cond x wf
  for w, wf in enumerate(['HF', 'LF']):
    wf_idx = data['wf'] == wf
    for c, cond in enumerate(['pure', 'mixed', 'long']):
      cond_idx = data['cond'] == cond
      for i in xrange(3):
        counts[:,4,c,w,i] = [np.sum(data[wf_idx & cond_idx & (data['type'] == 'lure') & (data['subj'] == subject)]['response'] == i) for subject in subjectList]
        counts[:,0,c,w,i] = [np.sum(data[wf_idx & cond_idx & (data['type'] == 'weak') & (data['source'] == 0) & (data['subj'] == subject)]['response'] == i) for subject in subjectList]
        counts[:,1,c,w,i] = [np.sum(data[wf_idx & cond_idx & (data['type'] == 'weak') & (data['source'] == 1) & (data['subj'] == subject)]['response'] == i) for subject in subjectList]
        if cond == 'mixed':
          counts[:,2,c,w,i] = [np.sum(data[wf_idx & cond_idx & (data['type'] == 'strong') & (data['source'] == 0) & (data['subj'] == subject)]['response'] == i) for subject in subjectList]
          counts[:,3,c,w,i] = [np.sum(data[wf_idx & cond_idx & (data['type'] == 'strong') & (data['source'] == 1) & (data['subj'] == subject)]['response'] == i) for subject in subjectList]

  #LENGTH/STRENGTH: sum over WF
  dataRates = np.zeros((nSubj,7,3))

  sm = np.sum(counts[:,0,:,:,0:2], axis = (-1,-2)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,0] = np.sum(counts[:,0,:,:,0], axis = -1) / sm

  sm = np.sum(counts[:,1,:,:,0:2], axis = (-1,-2)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,1] = np.sum(counts[:,1,:,:,1], axis = -1) / sm

  sm = np.sum(counts[:,2,:,:,0:2], axis = (-1,-2)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,2] = np.sum(counts[:,2,:,:,0], axis = -1) / sm

  sm = np.sum(counts[:,3,:,:,0:2], axis = (-1,-2)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,3] = np.sum(counts[:,3,:,:,1], axis = -1) / sm

  sm = np.sum(counts[:,4], axis = (-1,-2)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,4] = np.sum(counts[:,4,:,:,0] + counts[:,4,:,:,1], axis = -1) / sm #FAR = (source A + source B) / CR

  sm = np.sum(counts[:,0:2], axis = (1,-1,-2)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,5] = np.sum(counts[:,0:2,:,:,[0,1]], axis = (1,-1,-2)) / sm

  sm = np.sum(counts[:,2:4], axis = (1,-1,-2)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,6] = np.sum(counts[:,2:4,:,:,[0,1]], axis = (1,-1,-2)) / sm
 
  ratesL = np.mean(dataRates, axis = 0)
  errorsL = sem(dataRates, axis = 0)

  #WORD FREQUENCY: sum over LL/LS
  dataRates = np.zeros((nSubj,7,2))

  sm = np.sum(counts[:,0,:,:,0:2], axis = (-1,-3)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,0] = np.sum(counts[:,0,:,:,0], axis = -2) / sm

  sm = np.sum(counts[:,1,:,:,0:2], axis = (-1,-3)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,1] = np.sum(counts[:,1,:,:,1], axis = -2) / sm

  sm = np.sum(counts[:,2,:,:,0:2], axis = (-1,-3)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,2] = np.sum(counts[:,2,:,:,0], axis = -2) / sm

  sm = np.sum(counts[:,3,:,:,0:2], axis = (-1,-3)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,3] = np.sum(counts[:,3,:,:,1], axis = -2) / sm

  sm = np.sum(counts[:,4], axis = (-1,-3)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,4] = np.sum(counts[:,4,:,:,0] + counts[:,4,:,:,1], axis = -2) / sm #FAR = (source A + source B) / CR

  sm = np.sum(counts[:,0:2], axis = (1,-1,-3)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,5] = np.sum(counts[:,0:2,:,:,[0,1]], axis = (1,-1,-3)) / sm

  sm = np.sum(counts[:,2:4], axis = (1,-1,-3)).astype(np.float32)
  sm[sm == 0] = 1
  dataRates[:,6] = np.sum(counts[:,2:4,:,:,[0,1]], axis = (1,-1,-3)) / sm
 
  ratesW = np.nanmean(dataRates, axis = 0)
  errorsW = sem(dataRates, axis = 0)
    
  return ratesL, errorsL, ratesW, errorsW

from hpd import HDI_from_MCMC
def noiseSources(fit = 'sourceDev4COCOfreq', fontsize = 8, nCol = 1, col = 0):
  
  nRows = 2

  theta = getPhiMu(fit)
  p = getParams(fit)
  nChains, nmc, nParams = theta.shape
  theta = theta.reshape(nChains*nmc,nParams)
  nChains = nChains * nmc

  dataset = dataName(fit)
  data = returnData(dataset)
  
  if 'Vss' in p: Vss = theta[:,p.index('Vss')]
  elif 'logVss' in p: Vss = np.exp(theta[:,p.index('logVss')])
  else: Vss = .1
  if 'Vtt' in p: Vtt = theta[:,p.index('Vtt')]
  elif 'logVtt' in p: Vtt = np.exp(theta[:,p.index('logVtt')])
  else: Vtt = .02
 
  #n calculation
  age = np.mean(np.unique(data['age'])) #average age
  n = age * 10950000
  nmillion = n / 1000000

  #fixed parameters
  r = np.ones(nChains)
  rStrong = r + np.exp(theta[:,p.index('logrStrong')])

  Usu = 0
  Uti = 0
  Utt = 1.0
  Uss = 1.0

  Ucc = 1.0 #source match fixed to 1.0
  if 'Uaa' not in p: UaaList = 1.0
  else: UaaList = theta[:,p.index('Uaa')]
  UbbList = UaaList
  Uaa, Ubb = 1.0, 1.0
  Uab = 0.0
  Uba = 0.0
  Vti = np.exp(theta[:,p.index('logVti')])
  Vsu = np.exp(theta[:,p.index('logVsu')])

  Vaa = np.exp(theta[:,p.index('logVaa')])
  if 'logVab' in p: Vab = np.exp(theta[:,p.index('logVab')])
  else: Vab = Vaa
  Vbb = Vaa
  Vba = Vab
  if 'logVcc' in p: Vcc = np.exp(theta[:,p.index('logVcc')])
  else: Vcc = 0

  Uac = 0
  Ubc = 0
  
  if 'logVac' in p: Vac = np.exp(theta[:,p.index('logVac')])
  else: Vac = 0
  Vbc = Vac

  if 'nSourceProp' in p: nSourceProp = theta[:,p.index('nSourceProp')]
  else: nSourceProp = np.ones(nChains) * .10
  if 'mSourceProp' in p: mSourceProp = theta[:,p.index('mSourceProp')]
  else: mSourceProp = nSourceProp

  nSource = np.log(n * nSourceProp)
  n2 = np.log(n - nSource)
  n = np.log(n)

  ### ITEM RECOGNITION ###

  selfMatch = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vcc + Ucc**2)) - (Utt**2 * Uss**2 * Ucc**2)
  matchOthers = ((Vti + Uti**2) * (Vss + Uss**2) * (Vcc + Ucc**2)) - (Uti**2 * Uss**2 * Ucc**2)
  contextNoise = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vcc + Ucc**2)) - (Utt**2 * Usu**2 * Ucc**2)
  bgNoise = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vcc + Ucc**2)) - (Uti**2 * Usu**2 * Ucc**2)
  
  mLF = np.mean(np.log(data[data['wf'] == 'LF']['ageFreq'] * nmillion))
  mHF = np.mean(np.log(data[data['wf'] == 'HF']['ageFreq'] * nmillion))

  #interference sources for item recognition
  BN = n * bgNoise
  CNLF = mLF * contextNoise
  CNHF = mHF * contextNoise
  INshort = 8 * matchOthers
  INlong = 16 * matchOthers
  INmixed = (8 * matchOthers) + ((8 * (rStrong)) * matchOthers)

  sources = [INshort, INlong, INmixed, CNLF, CNHF, BN]
  labels = ['Item\nShort', 'Item\nLong', 'Item\nMixed', 'Context\nLF', 'Context\nHF', 'Background']

  subplot(nRows,nCol,col + 1)

  for i in xrange(len(sources)):
    mn = np.mean(sources[i])
    lower, upper = HDI_from_MCMC(sources[i], .95)
    error = np.array([mn - lower, upper - mn]).reshape(1,2)
    bar(i, mn, yerr = error, align = 'center', color = 'orange', edgecolor = 'black')
  ylabel('Variance', fontsize = fontsize)
  xticks(np.arange(len(sources)), labels, fontsize = fontsize)
  yticks(fontsize = fontsize)

  ### SOURCE MEMORY

  mSourceProp = mSourceProp.reshape(1,nChains)

  LFcounts = data[data['wf'] == 'LF']['ageFreq'] * nmillion
  LFcounts = LFcounts.reshape(len(LFcounts),1)
  HFcounts = data[data['wf'] == 'HF']['ageFreq'] * nmillion
  HFcounts = HFcounts.reshape(len(HFcounts),1)

  mSourceLF = np.mean(np.log(LFcounts * mSourceProp))
  mSourceHF = np.mean(np.log(HFcounts * mSourceProp))
  m2LF = np.mean(np.log(LFcounts - (LFcounts * mSourceProp)))
  m2HF = np.mean(np.log(HFcounts - (HFcounts * mSourceProp)))

  matchOthersAA = ((Vti + Uti**2) * (Vss + Uss**2) * (Vaa + Uaa**2)) - (Uti**2 * Uss**2 * UaaList**2)
  matchOthersAB = ((Vti + Uti**2) * (Vss + Uss**2) * (Vab + Uab**2)) - (Uti**2 * Uss**2 * Uab**2)
  matchOthersBB = ((Vti + Uti**2) * (Vss + Uss**2) * (Vbb + Ubb**2)) - (Uti**2 * Uss**2 * UbbList**2)
  matchOthersBA = ((Vti + Uti**2) * (Vss + Uss**2) * (Vba + Uba**2)) - (Uti**2 * Uss**2 * Uba**2)
  contextNoiseAA = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vaa + Uaa**2)) - (Utt**2 * Usu**2 * Uaa**2)
  contextNoiseAB = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vab + Uab**2)) - (Utt**2 * Usu**2 * Uab**2)
  contextNoiseBB = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vbb + Ubb**2)) - (Utt**2 * Usu**2 * Ubb**2)
  contextNoiseBA = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vba + Uba**2)) - (Utt**2 * Usu**2 * Uba**2)
  contextNoiseAC = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vac + Uac**2)) - (Utt**2 * Usu**2 * Uac**2) #item level context noise
  contextNoiseBC = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vac + Ubc**2)) - (Utt**2 * Usu**2 * Ubc**2) #item level context noise
  bgNoiseAA = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vaa + Uaa**2)) - (Vti**2 * Vsu**2 * Vaa**2)
  bgNoiseAB = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vab + Uab**2)) - (Vti**2 * Vsu**2 * Vab**2)
  bgNoiseBB = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vaa + Uaa**2)) - (Vti**2 * Vsu**2 * Vaa**2)
  bgNoiseBA = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vba + Uba**2)) - (Vti**2 * Vsu**2 * Vba**2)
  bgNoiseAC = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vac + Uac**2)) - (Vti**2 * Vsu**2 * Vac**2)
  bgNoiseBC = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vbc + Ubc**2)) - (Vti**2 * Vsu**2 * Vbc**2)

  INshort = (4 * matchOthersAA) + (4 * matchOthersAB)
  INlong = (8 * matchOthersAA) + (8 * matchOthersAB)
  INmixed = (2 * matchOthersAA) + (2 * matchOthersAB) + ((2 * rStrong**2) * matchOthersAA) + ((2 * rStrong*82) * matchOthersAB)

  BN_SMatch = nSource * bgNoiseAA
  BN_SMismatch = nSource * bgNoiseAB
  BN_SOther = n2 * bgNoiseAC
  BN = BN_SMatch + BN_SMismatch + BN_SOther

  CNLF_SMatch = mSourceLF * contextNoiseAA
  CNLF_SMismatch = mSourceLF * contextNoiseAB
  CNLF_SOther = m2LF * contextNoiseAC
  CNLF = CNLF_SMatch + CNLF_SMismatch + CNLF_SOther

  CNHF_SMatch = mSourceHF * contextNoiseAA
  CNHF_SMismatch = mSourceHF * contextNoiseAB
  CNHF_SOther = m2HF * contextNoiseAC
  CNHF = CNHF_SMatch + CNHF_SMismatch + CNHF_SOther
 
  #sources = [INshort, INlong, INmixed, CNLF_SMatch, CNLF_SMismatch, CNLF_SOther, CNHF_SMatch, CNHF_SMismatch, CNHF_SOther, BN_SMatch, BN_SMismatch, BN_SOther]
  #labels = ['Item\nShort', 'Item\nLong', 'Item\nMixed', 'Context\nLF\nMatch', 'Context\nLF\nMismatch', 'Context\nLF\nOther', 'Context\nHF\nMatch', 'Context\nHF\nMismatch', 'Context\nHF\nOther]', 'Background\nMatch', 'Background\nMismatch', 'Background\nOther']

  sources = [INshort, INlong, INmixed, CNLF, CNHF, BN]
  labels = ['Item\nShort', 'Item\nLong', 'Item\nMixed', 'Context\nLF', 'Context\nHF', 'Background']

  subplot(nRows,nCol,col + 1 + nCol)

  for i in xrange(len(sources)):
    mn = np.mean(sources[i])
    lower, upper = HDI_from_MCMC(sources[i], .95)
    error = np.array([mn - lower, upper - mn]).reshape(1,2)
    bar(i, mn, yerr = error, align = 'center', color = 'orange', edgecolor = 'black')
  ylabel('Variance', fontsize = fontsize)
  xticks(np.arange(len(sources)), labels, fontsize = fontsize)
  yticks(fontsize = fontsize)

def noiseSourcesAges(version = ''):
 
  figure(figsize=(13,5.0))
  subplots_adjust(left = .05, bottom = .13, right = .95, top = .9, hspace = .37, wspace = .11)

  fits = ['sourceDev45COCOfreq' + version, 'sourceDev78COCOfreq' + version, 'sourceDevAdultsCOCOfreq' + version]
  labels = ['Age 4-5', 'Age 7-8', 'Adults']
  for f, fit in enumerate(fits):
    noiseSources(fit, nCol = 3, col = f, fontsize = 7)
    subplot(2,3,f+1)
    title(labels[f], fontsize = 11)
    ylim(-.025, 1.01)
    subplot(2,3,f+4)
    ylim(-.05, 1.01)
    if f != 0:
      subplot(2,3,f+1)
      #yticks([], [])
      ylabel('')
      subplot(2,3,f+4)
      #yticks([], [])
      ylabel('')
  tight_layout()
  
def savePlots(version = ''):

  ioff()
  
  noiseSourcesAges(version = version)
  savefig('devPlots/noiseSourcesAges' + version + '.png')
  clf()
  
  plotRatesAllMarginal(version = version)
  savefig('devPlots/ratesAllMarginal' + version + '.png')
  clf()
  
  compareAgesAll(version = version)
  savefig('devPlots/compareAgesAll' + version + '.png')
  clf()
  
  '''
  fits = ['sourceDev4COCOfreq', 'sourceDev8COCOfreq', 'sourceDevAdultsCOCOFreq']
  ages = ['4', '8', 'Adults']
  for f, fit in enumerate(fits):
    phiMu = getPhiMu(fit)
    p = getParams(fit)
    allTracePlots(phiMu, p)
    savefig('devPlots/trace_' + ages[f]  + version + '.png')
    clf()'''

def plotRecovery(age = '4', version = 'Vaa', fontsize = 8):
  if 'rBeta' in version: recVer = 'rBeta'
  elif 'Vaa2' in version: recVer = 'Vaa2'
  else: recVer = ''
  fit = 'sourceDev' + age + 'COCOfreq' + version
  rec = 'sourceDev' + age + 'Rec' + recVer + 'COCOfreq' + version

  colors = ['black', 'red']
  p = getParams(fit)

  graphs = []
  phiMu = getPhiMu(fit)
  phiMuRec = getPhiMu(rec)
  p = getParams(fit)
  for i, param in enumerate(p):
    subplot(3,3,i+1)
    vals = phiMu[:,:,p.index(param)]
    valsRec = phiMuRec[:,:,p.index(param)]
    if i == 0:
      graphs.append(plotPosterior(vals, returnGraph = True, color = colors[0]))
      graphs.append(plotPosterior(valsRec, returnGraph = True, color = colors[1]))
    else:
      plotPosterior(vals, color = colors[0])
      plotPosterior(valsRec, color = colors[1])
    label = getLabel(param)
    title(label, fontsize = fontsize * 1.5)
    yticks([], [])
    xticks(fontsize = fontsize)
    
    if f == 2: legend(graphs, labels, loc = 'upper left', prop = legendFont)
  tight_layout()

def writeThetaMeans():
  data = returnData('sourceDev4Ex')
  subjectList = np.unique(data['subj'])
  f = open('thetaMeans.txt', 'w')
  p = getParams('sourceDev4ExCOCOfreqVaa_rBeta')
  
  #create the header
  sstr = 'Subject'
  for param in p:
    sstr += '\t' + param
  print >> f, sstr

  #add in the subject values
  for s, subject in enumerate(subjectList):
    theta = getSubj(s, 'sourceDev4ExCOCOfreqVaa_rBeta')
    sstr = str(subject)
    for i, param in enumerate(p):
      sstr += '\t' + str(np.mean(theta[:,:,p.index(param)]))
    print >> f, sstr
  f.close()

def noiseSourcesSubj(theta, p, data):
 
  nChains, nmc, nParams = theta.shape
  theta = theta.reshape(nChains*nmc,nParams)
  nChains = nChains * nmc
  
  if 'Vss' in p: Vss = theta[:,p.index('Vss')]
  elif 'logVss' in p: Vss = np.exp(theta[:,p.index('logVss')])
  else: Vss = .1
  if 'Vtt' in p: Vtt = theta[:,p.index('Vtt')]
  elif 'logVtt' in p: Vtt = np.exp(theta[:,p.index('logVtt')])
  else: Vtt = .02
 
  #n calculation
  age = np.mean(np.unique(data['age'])) #average age
  n = age * 10950000
  nmillion = n / 1000000

  #fixed parameters
  r = np.ones(nChains)
  rStrong = r + np.exp(theta[:,p.index('logrStrong')])

  Usu = 0
  Uti = 0
  Utt = 1.0
  Uss = 1.0

  Ucc = 1.0 #source match fixed to 1.0
  if 'Uaa' not in p: UaaList = 1.0
  else: UaaList = theta[:,p.index('Uaa')]
  UbbList = UaaList
  Uaa, Ubb = 1.0, 1.0
  Uab = 0.0
  Uba = 0.0
  Vti = np.exp(theta[:,p.index('logVti')])
  Vsu = np.exp(theta[:,p.index('logVsu')])

  Vaa = np.exp(theta[:,p.index('logVaa')])
  if 'logVab' in p: Vab = np.exp(theta[:,p.index('logVab')])
  else: Vab = Vaa
  Vbb = Vaa
  Vba = Vab
  if 'logVcc' in p: Vcc = np.exp(theta[:,p.index('logVcc')])
  else: Vcc = 0

  Uac = 0
  Ubc = 0
  
  if 'logVac' in p: Vac = np.exp(theta[:,p.index('logVac')])
  else: Vac = 0
  Vbc = Vac

  if 'nSourceProp' in p: nSourceProp = theta[:,p.index('nSourceProp')]
  else: nSourceProp = np.ones(nChains) * .10
  if 'mSourceProp' in p: mSourceProp = theta[:,p.index('mSourceProp')]
  else: mSourceProp = nSourceProp

  nSource = np.log(n * nSourceProp)
  n2 = np.log(n - nSource)
  n = np.log(n)

  ### ITEM RECOGNITION ###

  selfMatch = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vcc + Ucc**2)) - (Utt**2 * Uss**2 * Ucc**2)
  matchOthers = ((Vti + Uti**2) * (Vss + Uss**2) * (Vcc + Ucc**2)) - (Uti**2 * Uss**2 * Ucc**2)
  contextNoise = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vcc + Ucc**2)) - (Utt**2 * Usu**2 * Ucc**2)
  bgNoise = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vcc + Ucc**2)) - (Uti**2 * Usu**2 * Ucc**2)
  
  mLF = np.mean(np.log(data[data['wf'] == 'LF']['ageFreq'] * nmillion))
  mHF = np.mean(np.log(data[data['wf'] == 'HF']['ageFreq'] * nmillion))

  #interference sources for item recognition
  BN = n * bgNoise
  CNLF = mLF * contextNoise
  CNHF = mHF * contextNoise
  INshort = 8 * matchOthers
  INlong = 16 * matchOthers
  INmixed = (8 * matchOthers) + ((8 * (rStrong)) * matchOthers)

  sourcesItem = [INshort, INlong, INmixed, CNLF, CNHF, BN]
  sourcesItem = [np.mean(s) for s in sourcesItem]

  ### SOURCE MEMORY

  mSourceProp = mSourceProp.reshape(1,nChains)

  LFcounts = data[data['wf'] == 'LF']['ageFreq'] * nmillion
  LFcounts = LFcounts.reshape(len(LFcounts),1)
  HFcounts = data[data['wf'] == 'HF']['ageFreq'] * nmillion
  HFcounts = HFcounts.reshape(len(HFcounts),1)

  mSourceLF = np.mean(np.log(LFcounts * mSourceProp))
  mSourceHF = np.mean(np.log(HFcounts * mSourceProp))
  m2LF = np.mean(np.log(LFcounts - (LFcounts * mSourceProp)))
  m2HF = np.mean(np.log(HFcounts - (HFcounts * mSourceProp)))

  matchOthersAA = ((Vti + Uti**2) * (Vss + Uss**2) * (Vaa + Uaa**2)) - (Uti**2 * Uss**2 * UaaList**2)
  matchOthersAB = ((Vti + Uti**2) * (Vss + Uss**2) * (Vab + Uab**2)) - (Uti**2 * Uss**2 * Uab**2)
  matchOthersBB = ((Vti + Uti**2) * (Vss + Uss**2) * (Vbb + Ubb**2)) - (Uti**2 * Uss**2 * UbbList**2)
  matchOthersBA = ((Vti + Uti**2) * (Vss + Uss**2) * (Vba + Uba**2)) - (Uti**2 * Uss**2 * Uba**2)
  contextNoiseAA = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vaa + Uaa**2)) - (Utt**2 * Usu**2 * Uaa**2)
  contextNoiseAB = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vab + Uab**2)) - (Utt**2 * Usu**2 * Uab**2)
  contextNoiseBB = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vbb + Ubb**2)) - (Utt**2 * Usu**2 * Ubb**2)
  contextNoiseBA = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vba + Uba**2)) - (Utt**2 * Usu**2 * Uba**2)
  contextNoiseAC = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vac + Uac**2)) - (Utt**2 * Usu**2 * Uac**2) #item level context noise
  contextNoiseBC = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vac + Ubc**2)) - (Utt**2 * Usu**2 * Ubc**2) #item level context noise
  bgNoiseAA = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vaa + Uaa**2)) - (Vti**2 * Vsu**2 * Vaa**2)
  bgNoiseAB = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vab + Uab**2)) - (Vti**2 * Vsu**2 * Vab**2)
  bgNoiseBB = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vaa + Uaa**2)) - (Vti**2 * Vsu**2 * Vaa**2)
  bgNoiseBA = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vba + Uba**2)) - (Vti**2 * Vsu**2 * Vba**2)
  bgNoiseAC = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vac + Uac**2)) - (Vti**2 * Vsu**2 * Vac**2)
  bgNoiseBC = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vbc + Ubc**2)) - (Vti**2 * Vsu**2 * Vbc**2)

  INshort = (4 * matchOthersAA) + (4 * matchOthersAB)
  INlong = (8 * matchOthersAA) + (8 * matchOthersAB)
  INmixed = (2 * matchOthersAA) + (2 * matchOthersAB) + ((2 * rStrong**2) * matchOthersAA) + ((2 * rStrong*82) * matchOthersAB)

  BN_SMatch = nSource * bgNoiseAA
  BN_SMismatch = nSource * bgNoiseAB
  BN_SOther = n2 * bgNoiseAC
  BN = BN_SMatch + BN_SMismatch + BN_SOther

  CNLF_SMatch = mSourceLF * contextNoiseAA
  CNLF_SMismatch = mSourceLF * contextNoiseAB
  CNLF_SOther = m2LF * contextNoiseAC
  CNLF = CNLF_SMatch + CNLF_SMismatch + CNLF_SOther

  CNHF_SMatch = mSourceHF * contextNoiseAA
  CNHF_SMismatch = mSourceHF * contextNoiseAB
  CNHF_SOther = m2HF * contextNoiseAC
  CNHF = CNHF_SMatch + CNHF_SMismatch + CNHF_SOther

  sourcesSource = [INshort, INlong, INmixed, CNLF, CNHF, BN]
  sourcesSource = [np.mean(s) for s in sourcesSource]

  return sourcesItem, sourcesSource

def writeNoiseSources():
  data = returnData('sourceDev4Ex')
  subjectList = np.unique(data['subj'])
  f = open('sourcesMeans.txt', 'w')
  p = getParams('sourceDev4ExCOCOfreqVaa_rBeta')

  #create the header
  sstr = 'Subject'
  labels = ['INshort_item', 'INlong_item', 'INmixed_item', 'CNLF_item', 'CNHF_item', 'BN_item', 'INshort_source', 'INlong_source', 'INmixed_source', 'CNLF_source', 'CNHF_source', 'BN_source']
  for label in labels:
    sstr += '\t' +label
  print >> f, sstr

  #add in the subject values
  for s, subject in enumerate(subjectList):
    theta = getSubj(s, 'sourceDev4ExCOCOfreqVaa_rBeta')
    sstr = str(subject)
    sourcesItem, sourcesSource = noiseSourcesSubj(theta, p, data[data['subj'] == subject])
    sources = sourcesItem + sourcesSource
    for i, source in enumerate(sources):
      sstr += '\t' + str(source)
    print >> f, sstr
  f.close()
  
  
