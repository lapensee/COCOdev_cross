### code for hierarchical differential evolution MCMC
### main functions: crossover for subject params, crossoverHyper for hyperparameters

### a note on truncated normal distributions in Python:
### the clip point a is with reference to a standard normal distribution: to convert, use (clip - mean) / scale

### some helpful reminders:

# truncnorm function couldn't be fully vectorized - wrote my new functions as dtnorm and rtnorm

# adjusted to be used with both normal and different types of truncnorm distributions

# cutoffs on RTs for RT models now depend on the dataset

# prior calculations now set anything non-finite to -inf. This prevents positive infs from being accepted

import numpy as np
np.seterr(all = 'ignore')
from scipy.stats import norm, truncnorm, gamma, uniform, lognorm, beta, mstats
from deFuncs import dtnorm, rtnorm, findNearest, twoNearest
import sys
sys.stdout.flush()
from joblib import Parallel, delayed, load, dump
import cPickle as pickle
import random, tempfile, time, os, copy, gc
from random import shuffle
from extract import *

szMax = .9

#log density of subject parameters under group level parameters
def log_dens_hyper(theta, phiMu, phiSigma, dist):

  nSubj,nChains,nParams = theta.shape
  dens = np.zeros((nSubj,nChains))

  sigma = phiSigma
  if 'precision' in dist.keys(): sigma = 1 / phiSigma
  elif 'logSigma' in dist.keys(): sigma = np.exp(phiSigma)
  if 'dtnorm' in dist.keys():
    dens += np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm']], 0, np.inf, loc = phiMu[:,dist['dtnorm']], scale = sigma[:,dist['dtnorm']])), axis = 2)
  if 'dtnorm01' in dist.keys():
    dens += np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm01']], 0, 1.0, loc = phiMu[:,dist['dtnorm01']], scale = sigma[:,dist['dtnorm01']])), axis = 2)
  if 'norm' in dist.keys():
    dens += np.sum(np.log(norm.pdf(theta[:,:,dist['norm']], loc = phiMu[:,dist['norm']], scale = sigma[:,dist['norm']])), axis = 2)
  if 'beta' in dist.keys():
    #reparameterization: mean and *sample size*
    a = phiMu[:,dist['beta']] * phiSigma[:,dist['beta']]
    b = (1 - phiMu[:,dist['beta']]) * phiSigma[:,dist['beta']]
    dens += np.sum(np.log(beta.pdf(theta[:,:,dist['beta']], a = a, b = b)), axis = 2)
    #don't allow values that are 1 or 0
    betaex = (np.prod(theta[:,:,dist['beta']] == 1.0, axis = 2) == 1) | (np.prod(theta[:,:,dist['beta']] == 0, axis = 2) == 1)
    dens[betaex] = -1 * np.inf
  dens[np.isfinite(dens) == False] = -1 * np.inf
  return dens

#just have to specify nSubj for the between subjects case
def log_dens_hyperBetween(thetas, phiMu, phiSigma, dists, distHiers, nSubj):

  nChains,nParams = phiMu.shape

  dens = np.zeros((nSubj,nChains))
  lower = 0

  sigma = phiSigma

  for g in xrange(len(thetas)):
    theta, dist, distHier = thetas[g], dists[g], distHiers[g]
    upper = lower + len(theta)

    if 'dtnorm' in dist.keys():
      dens[lower:upper] += np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm']], 0, np.inf, loc = phiMu[:,distHier['dtnorm']], scale = sigma[:,distHier['dtnorm']])), axis = 2)
    if 'dtnorm01' in dist.keys():
      dens[lower:upper] += np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm01']], 0, 1.0, loc = phiMu[:,distHier['dtnorm01']], scale = sigma[:,distHier['dtnorm01']])), axis = 2)
    if 'norm' in dist.keys():
      dens[lower:upper] += np.sum(np.log(norm.pdf(theta[:,:,dist['norm']], loc = phiMu[:,distHier['norm']], scale = sigma[:,distHier['norm']])), axis = 2)
    if ('beta' in dist.keys()) & (len(dist['beta']) > 0):
      #reparameterization: mean and *sample size*
      a = phiMu[:,distHier['beta']] * phiSigma[:,distHier['beta']]
      b = (1 - phiMu[:,distHier['beta']]) * phiSigma[:,distHier['beta']]
      dens[lower:upper] += np.sum(np.log(beta.pdf(theta[:,:,dist['beta']], a = a, b = b)), axis = 2)
      #don't allow values that are 1 or 0
      betaex = (np.prod(theta[:,:,dist['beta']] == 1.0, axis = 2) == 1) | (np.prod(theta[:,:,dist['beta']] == 0, axis = 2) == 1)
      dens[lower:upper][betaex] = -1 * np.inf
    lower = upper
  dens[np.isfinite(dens) == False] = -1 * np.inf
  return dens    

#keeps likelihood of each parameter separate
def log_dens_hyper_separate(theta, phiMu, phiSigma, dist):
  
  nSubj,nChains,nParams = theta.shape
  sigma = phiSigma
  dens = np.zeros((nSubj,nChains,nParams))
  if 'dtnorm' in dist.keys(): 
    dens[:,:,dist['dtnorm']] = np.log(dtnorm(theta[:,:,dist['dtnorm']], 0, np.inf, loc = phiMu[:,dist['dtnorm']], scale = sigma[:,dist['dtnorm']]))
  if 'dtnorm01' in dist.keys():
    dens[:,:,dist['dtnorm01']] = np.log(dtnorm(theta[:,:,dist['dtnorm01']], 0, 1.0, loc = phiMu[:,dist['dtnorm01']], scale = sigma[:,dist['dtnorm01']]))
  if 'dtnorm.25' in dist.keys():
    dens[:,:,dist['dtnorm.25']] = np.log(dtnorm(theta[:,:,dist['dtnorm.25']], 0, .25, loc = phiMu[:,dist['dtnorm.25']], scale = sigma[:,dist['dtnorm.25']]))
  if 'norm' in dist.keys():
    dens[:,:,dist['norm']] = np.log(norm.pdf(theta[:,:,dist['norm']], loc = phiMu[:,dist['norm']], scale = sigma[:,dist['norm']]))
  if 'beta' in dist.keys():
    #reparameterization
    a = phiMu[:,dist['beta']] * phiSigma[:,dist['beta']]**2
    b = (1 - phiMu[:,dist['beta']]) * phiSigma[:,dist['beta']]**2
    dens[:,:,dist['beta']] = np.log(beta.pdf(theta[:,:,dist['beta']], a = a, b = b))
    #don't allow values that are 1 or 0
    betaex = (np.prod(theta[:,:,dist['beta']] == 1.0, axis = 2) == 1) | (np.prod(theta[:,:,dist['beta']] == 0, axis = 2) == 1)
    dens[betaex] = -1 * np.inf
  return dens

#this is used for group/subject fits
def log_dens_prior(theta, priors, dist):
  priorMus, priorSigmas = priors
  nChains, nParams = theta.shape
  priorDens = np.zeros(nChains)
  if 'dtnorm' in dist.keys():
    priorDens += np.sum(np.log(dtnorm(theta[:,dist['dtnorm']], 0, np.inf, loc = priorMus[dist['dtnorm']], scale = priorSigmas[dist['dtnorm']])), axis = 1)
  if 'dtnorm01' in dist.keys():
    priorDens += np.sum(np.log(dtnorm(theta[:,dist['dtnorm01']], 0, 1.0, loc = priorMus[dist['dtnorm01']], scale = priorSigmas[dist['dtnorm01']])), axis = 1)
  if 'dtnorm.25' in dist.keys():
    priorDens += np.sum(np.log(dtnorm(theta[:,dist['dtnorm.25']], 0, .25, loc = priorMus[dist['dtnorm.25']], scale = priorSigmas[dist['dtnorm.25']])), axis = 1)
  if 'norm' in dist.keys():
    priorDens += np.sum(np.log(norm.pdf(theta[:,dist['norm']], loc = priorMus[dist['norm']], scale = priorSigmas[dist['norm']])), axis = 1)
  if 'beta' in dist.keys():
    a = priorMus[dist['beta']] * priorSigmas[dist['beta']]
    b = (1 - priorMus[dist['beta']]) * priorSigmas[dist['beta']]
    dens = np.sum(np.log(beta.pdf(theta[:,dist['beta']], a = a, b = b)), axis = 1)
    #don't allow values that are 1 or 0
    betaex = (np.prod(theta[:,dist['beta']] == 1.0, axis = 1) == 1) | (np.prod(theta[:,dist['beta']] == 0, axis = 1) == 1)
    dens[betaex] = -1 * np.inf
    dens[np.isinf(dens)] = -1 * np.inf
    priorDens += dens

  priorDens[np.isfinite(priorDens) == False] = -1 * np.inf
  return priorDens

#this is used for fitting multiple subjects non-hierarchically
#changes from above: added subject dimension to theta, sum over axis 2 instead of 1
def log_dens_priorSubjects(theta, priors, dist):
  priorMus, priorSigmas = priors
  nSubj, nChains, nParams = theta.shape
  priorDens = np.zeros((nSubj, nChains))
  if 'dtnorm' in dist.keys():
    priorDens += np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm']], 0, np.inf, loc = priorMus[dist['dtnorm']], scale = priorSigmas[dist['dtnorm']])), axis = 2)
  if 'dtnorm01' in dist.keys():
    priorDens += np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm01']], 0, 1.0, loc = priorMus[dist['dtnorm01']], scale = priorSigmas[dist['dtnorm01']])), axis = 2)
  if 'dtnorm.25' in dist.keys():
    priorDens += np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm.25']], 0, .25, loc = priorMus[dist['dtnorm.25']], scale = priorSigmas[dist['dtnorm.25']])), axis = 2)
  if 'norm' in dist.keys():
    priorDens += np.sum(np.log(norm.pdf(theta[:,:,dist['norm']], loc = priorMus[dist['norm']], scale = priorSigmas[dist['norm']])), axis = 2)
  if 'beta' in dist.keys():
    a = priorMus[dist['beta']] * priorSigmas[dist['beta']]
    b = (1 - priorMus[dist['beta']]) * priorSigmas[dist['beta']]
    dens = np.sum(np.log(beta.pdf(theta[:,:,dist['beta']], a = a, b = b)), axis = 2)
    #don't allow values that are 1 or 0
    betaex = (np.prod(theta[:,:,dist['beta']] == 1.0, axis = 2) == 1) | (np.prod(theta[:,:,dist['beta']] == 0, axis = 2) == 1)
    dens[betaex] = -1 * np.inf
    dens[np.isinf(dens)] = -1 * np.inf
    priorDens += dens

  priorDens[np.isfinite(priorDens) == False] = -1 * np.inf
  return priorDens

#this is used for DIC calculation
def log_dens_prior2(theta, priors, dist):
  priorMus, priorSigmas = priors
  nChains, nSamples, nParams = theta.shape
  priorDens = np.zeros((nChains,nSamples))
  if 'dtnorm' in dist.keys():
    priorDens += np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm']], 0, np.inf, loc = priorMus[dist['dtnorm']], scale = priorSigmas[dist['dtnorm']])), axis = 2)
  if 'dtnorm01' in dist.keys():
    priorDens += np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm01']], 0, 1.0, loc = priorMus[dist['dtnorm01']], scale = priorSigmas[dist['dtnorm01']])), axis = 2)
  if 'dtnorm.25' in dist.keys():
    priorDens += np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm.25']], 0, .25, loc = priorMus[dist['dtnorm.25']], scale = priorSigmas[dist['dtnorm.25']])), axis = 2)
  if 'norm' in dist.keys():
    priorDens += np.sum(np.log(norm.pdf(theta[:,:,dist['norm']], loc = priorMus[dist['norm']], scale = priorSigmas[dist['norm']])), axis = 2)
  return priorDens

#log density of all subjects parameters under group level parameters
#...KDE priors are carried in kwargs
def log_dens_hyper_and_prior(theta, phiMu, phiSigma, priors, dist, p, kdePriors = None):
  #construct list indices with the names of all the prior parameters
  priorMuMus, priorMuSigmas, priorSigmaAlphas, priorSigmaBetas = priors
  nChains,nParams = phiMu.shape
  subj, groupMu, groupSigma = np.zeros((nChains,nParams)), np.zeros((nChains,nParams)), np.zeros((nChains,nParams))

  sigma = phiSigma
  #if 'precision' in dist.keys(): sigma = 1 / np.sqrt(sigma)
  if 'precision' in dist.keys(): sigma = 1 / sigma
  if 'logSigma' in dist.keys(): sigma = np.exp(sigma)

  if 'dtnorm' in dist.keys():
    subj[:,dist['dtnorm']] = np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm']], 0, np.inf, loc = phiMu[:,dist['dtnorm']], scale = sigma[:,dist['dtnorm']])), axis = 0)
    groupMu[:,dist['dtnorm']] = np.log(dtnorm(phiMu[:,dist['dtnorm']], 0, np.inf, loc = priorMuMus[dist['dtnorm']], scale = priorMuSigmas[dist['dtnorm']]))

  if 'logSigma' not in dist.keys(): groupSigma = np.log(gamma.pdf(sigma, priorSigmaAlphas, scale = priorSigmaBetas))
  elif 'kde' not in dist.keys(): groupSigma = np.log(norm.pdf(np.log(sigma), priorSigmaAlphas, scale = priorSigmaBetas))
  else:
    idx = [prm for prm in xrange(nParams) if prm not in dist['kde']]
    groupSigma[idx] = np.log(norm.pdf(np.log(sigma), priorSigmaAlphas, scale = priorSigmaBetas))

  if 'dtnorm01' in dist.keys():
    subj[:,dist['dtnorm01']] = np.sum(np.log(dtnorm(theta[:,:,dist['dtnorm01']], 0, 1.0, loc = phiMu[:,dist['dtnorm01']], scale = sigma[:,dist['dtnorm01']])), axis = 0)
    groupMu[:,dist['dtnorm01']] = np.log(dtnorm(phiMu[:,dist['dtnorm01']], 0, 1.0, loc = priorMuMus[dist['dtnorm01']], scale = priorMuSigmas[dist['dtnorm01']]))
  if 'norm' in dist.keys():
    subj[:,dist['norm']] = np.sum(np.log(norm.pdf(theta[:,:,dist['norm']], loc = phiMu[:,dist['norm']], scale = sigma[:,dist['norm']])), axis = 0)
    groupMu[:,dist['norm']] = np.log(norm.pdf(phiMu[:,dist['norm']], loc = priorMuMus[dist['norm']], scale = priorMuSigmas[dist['norm']]))
  if 'beta' in dist.keys():
    if len(dist['beta']) > 0:
      #reparameterization
      a = phiMu[:,dist['beta']] * phiSigma[:,dist['beta']]
      b = (1 - phiMu[:,dist['beta']]) * phiSigma[:,dist['beta']]
      subj[:,dist['beta']] = np.sum(np.log(beta.pdf(theta[:,:,dist['beta']], a = a, b = b)), axis = 0)
      #don't allow values that are 1 or 0
      betaex = (np.prod(phiMu[:,dist['beta']] == 1.0, axis = 1) == 1) | (np.prod(phiMu[:,dist['beta']] == 0, axis = 1) == 1)
      subj[betaex] = -1 * np.inf
      a = priorMuMus[dist['beta']] * priorMuSigmas[dist['beta']]
      b = (1 - priorMuMus[dist['beta']]) * priorMuSigmas[dist['beta']]
      groupMu[:,dist['beta']] = np.log(beta.pdf(phiMu[:,dist['beta']], a = a, b = b))
      #don't allow values that are 1 or 0
      groupMu[betaex] = -1 * np.inf
  #kde: this allows for informed priors that were estimated using kernel density estimation
  #...note that this captures the PRIOR only - the subjects are captured above
  if 'kde' in dist.keys():
    for prm in dist['kde']: #search by parameter index
      param = p[prm] 
      vals = phiMu[:,prm]
      exclude = (vals < min(kdePriors['mu'][param]['support'])) | (vals > max(kdePriors['mu'][param]['support']))
      excluded = np.where(exclude)[0]
      groupMu[excluded,prm] = -1 * np.inf #parameters outside the range of the prior are automatically excluded

      if len(excluded) != nChains:
        include = exclude == False
        included = np.where(include)[0]
        support, density = kdePriors['mu'][param]['support'], kdePriors['mu'][param]['density']
        '''Method below just finds the nearest value'''
        #includedVals = np.array([findNearest(val, kdePriors['mu'][param]['support']) for val in vals[included]]) #find nearest values in the KDE
        #includedIdx = np.array([np.where(kdePriors['mu'][param]['support'] == val)[0][0] for val in includedVals])
        #groupMu[included,prm] = np.log(np.array([kdePriors['mu'][param]['density'][idx] for idx in includedIdx])) 
        groupMu[included,prm] = np.log(linear_interpolation(vals[included], support, density))

      vals = phiSigma[:,prm]
      exclude = (vals < min(kdePriors['sigma'][param]['support'])) | (vals > max(kdePriors['sigma'][param]['support']))
      excluded = np.where(exclude)[0]
      groupSigma[excluded,prm] = -1 * np.inf #parameters outside the range of the prior are automatically excluded

      if len(excluded) != nChains:
        include = exclude == False
        included = np.where(include)[0]
        support, density = kdePriors['sigma'][param]['support'], kdePriors['sigma'][param]['density']
        #includedVals = np.array([findNearest(val, kdePriors['sigma'][param]['support']) for val in vals[included]]) #find nearest values in the KDE
        #includedIdx = np.array([np.where(kdePriors['sigma'][param]['support'] == val)[0][0] for val in includedVals])
        #logdens = np.log(np.array([kdePriors['sigma'][param]['density'][idx] for idx in includedIdx]))
        #groupSigma[included,prm] = np.log(np.array([kdePriors['sigma'][param]['density'][idx] for idx in includedIdx]))
        groupSigma[included,prm] = np.log(linear_interpolation(vals[included], support, density))

  subj[np.isfinite(subj) == False] = -1 * np.inf
  groupMu[np.isfinite(groupMu) == False] = -1 * np.inf
  groupSigma[np.isfinite(groupSigma) == False] = -1 * np.inf
  return subj + groupMu + groupSigma #keep params intact, just sum over subjects

#linear interpolation for the KDE above
def linear_interpolation(vals, support, density):
  idx = np.array([twoNearest(val, support) for val in vals]) #find two nearest values in support
  m = np.array([(density[i1] - density[i0]) / (support[i1] - support[i0]) for i0, i1 in idx])
  b = np.array([density[idx[i][0]] - (m[i] * support[idx[i][0]]) for i in xrange(len(idx))])
  return m * vals + b 

#chains is passed in below - hard to swap chains in theta given the list structure
def log_dens_hyper_and_priorBetween(thetas, phiMu, phiSigma, priors, dists, distHiers, distHierAll, hp, chains = -1, kdePriors = None):
  #construct list indices with the names of all the prior parameters
  priorMuMus, priorMuSigmas, priorSigmaAlphas, priorSigmaBetas = priors
  nChains,nParams = phiMu.shape
  if chains == -1: chains = range(nChains)
  subj, groupMu, groupSigma = np.zeros((nChains,nParams)), np.zeros((nChains,nParams)), np.zeros((nChains,nParams))

  sigma = phiSigma

  if 'dtnorm' in distHierAll.keys():
    for g in xrange(len(thetas)):
      theta, dist, distHier = thetas[g], dists[g], distHiers[g]
      if len(dist['dtnorm']) > 0:
        subj[:,distHier['dtnorm']] += np.sum(np.log(dtnorm(theta[:,chains][:,:,dist['dtnorm']], 0, np.inf, loc = phiMu[:,distHier['dtnorm']], scale = sigma[:,distHier['dtnorm']])), axis = 0)
    groupMu[:,distHierAll['dtnorm']] = np.log(dtnorm(phiMu[:,distHierAll['dtnorm']], 0, np.inf, loc = priorMuMus[distHierAll['dtnorm']], scale = priorMuSigmas[distHierAll['dtnorm']]))

  if 'kde' not in distHierAll.keys(): groupSigma = np.log(gamma.pdf(sigma, priorSigmaAlphas, scale = priorSigmaBetas))
  else:
    idx = [prm for prm in xrange(nParams) if prm not in dist['kde']]
    groupSigma[idx] = np.log(norm.pdf(np.log(sigma), priorSigmaAlphas, scale = priorSigmaBetas))

  if 'dtnorm01' in distHierAll.keys():
    for g in xrange(len(thetas)):
      theta, dist, distHier = thetas[g], dists[g], distHiers[g]
      if len(dist['dtnorm01']) > 0:
        subj[:,distHier['dtnorm01']] += np.sum(np.log(dtnorm(theta[:,chains][:,:,dist['dtnorm01']], 0, 1.0, loc = phiMu[:,distHier['dtnorm01']], scale = sigma[:,distHier['dtnorm01']])), axis = 0)
    groupMu[:,distHierAll['dtnorm01']] = np.log(dtnorm(phiMu[:,distHierAll['dtnorm01']], 0, 1.0, loc = priorMuMus[distHierAll['dtnorm01']], scale = priorMuSigmas[distHierAll['dtnorm01']]))
  if 'norm' in distHierAll.keys():
    for g in xrange(len(thetas)):
      theta, dist, distHier = thetas[g], dists[g], distHiers[g]
      subj[:,distHier['norm']] += np.sum(np.log(norm.pdf(theta[:,chains][:,:,dist['norm']], loc = phiMu[:,distHier['norm']], scale = sigma[:,distHier['norm']])), axis = 0)
    groupMu[:,distHierAll['norm']] = np.log(norm.pdf(phiMu[:,distHierAll['norm']], loc = priorMuMus[distHierAll['norm']], scale = priorMuSigmas[distHierAll['norm']]))
  if 'beta' in distHierAll.keys():
    if len(distHierAll['beta']) > 0:
      for g in xrange(len(thetas)):
        theta, dist, distHier = thetas[g], dists[g], distHiers[g]
        #reparameterization
        if len(dist['beta']) > 0:
          a = phiMu[:,distHier['beta']] * phiSigma[:,distHier['beta']]
          b = (1 - phiMu[:,distHier['beta']]) * phiSigma[:,distHier['beta']]
          subj[:,distHier['beta']] += np.sum(np.log(beta.pdf(theta[:,chains][:,:,dist['beta']], a = a, b = b)), axis = 0)
      #don't allow values that are 1 or 0
      betaex = (np.prod(phiMu[:,distHierAll['beta']] == 1.0, axis = 1) == 1) | (np.prod(phiMu[:,distHierAll['beta']] == 0, axis = 1) == 1)
      subj[betaex] = -1 * np.inf
      a = priorMuMus[dist['beta']] * priorMuSigmas[dist['beta']]
      b = (1 - priorMuMus[dist['beta']]) * priorMuSigmas[dist['beta']]
      groupMu[:,distHierAll['beta']] = np.log(beta.pdf(phiMu[:,distHierAll['beta']], a = a, b = b))
      #don't allow values that are 1 or 0
      groupMu[betaex] = -1 * np.inf
  #kde: this allows for informed priors that were estimated using kernel density estimation
  #...note that this captures the PRIOR only - the subjects are captured above
  if 'kde' in dist.keys():
    for prm in dist['kde']: #search by parameter index
      param = hp[prm] 
      vals = phiMu[:,prm]
      exclude = (vals < min(kdePriors['mu'][param]['support'])) | (vals > max(kdePriors['mu'][param]['support']))
      excluded = np.where(exclude)[0]
      groupMu[excluded,prm] = -1 * np.inf #parameters outside the range of the prior are automatically excluded

      if len(excluded) != nChains:
        include = exclude == False
        included = np.where(include)[0]
        support, density = kdePriors['mu'][param]['support'], kdePriors['mu'][param]['density']
        '''Method below just finds the nearest value'''
        #includedVals = np.array([findNearest(val, kdePriors['mu'][param]['support']) for val in vals[included]]) #find nearest values in the KDE
        #includedIdx = np.array([np.where(kdePriors['mu'][param]['support'] == val)[0][0] for val in includedVals])
        #groupMu[included,prm] = np.log(np.array([kdePriors['mu'][param]['density'][idx] for idx in includedIdx])) 
        groupMu[included,prm] = np.log(linear_interpolation(vals[included], support, density))

      vals = phiSigma[:,prm]
      exclude = (vals < min(kdePriors['sigma'][param]['support'])) | (vals > max(kdePriors['sigma'][param]['support']))
      excluded = np.where(exclude)[0]
      groupSigma[excluded,prm] = -1 * np.inf #parameters outside the range of the prior are automatically excluded

      if len(excluded) != nChains:
        include = exclude == False
        included = np.where(include)[0]
        support, density = kdePriors['sigma'][param]['support'], kdePriors['sigma'][param]['density']
        #includedVals = np.array([findNearest(val, kdePriors['sigma'][param]['support']) for val in vals[included]]) #find nearest values in the KDE
        #includedIdx = np.array([np.where(kdePriors['sigma'][param]['support'] == val)[0][0] for val in includedVals])
        #logdens = np.log(np.array([kdePriors['sigma'][param]['density'][idx] for idx in includedIdx]))
        #groupSigma[included,prm] = np.log(np.array([kdePriors['sigma'][param]['density'][idx] for idx in includedIdx]))
        groupSigma[included,prm] = np.log(linear_interpolation(vals[included], support, density))

  subj[np.isfinite(subj) == False] = -1 * np.inf
  groupMu[np.isfinite(groupMu) == False] = -1 * np.inf
  groupSigma[np.isfinite(groupSigma) == False] = -1 * np.inf
  return subj + groupMu + groupSigma #keep params intact, just sum over subjects

#single parameter version: this is used for hypermigration
def log_dens_hyper_and_prior_singleParam(theta, phiMu, phiSigma, priors, dist, param, kdePriorsMu = None, kdePriorsSigma = None):
  
  #construct list indices with the names of all the prior parameters
  priorMuMus, priorMuSigmas, priorSigmaAlphas, priorSigmaBetas = priors

  #if 'precision' in dist.keys(): sigma = 1 / np.sqrt(sigma)
  if 'precision' in dist.keys(): sigma = 1 / phiSigma
  elif 'logSigma' in dist.keys(): sigma = np.exp(phiSigma)
  else: sigma = phiSigma

  if 'logSigma' not in dist.keys(): groupSigma = np.log(gamma.pdf(sigma, priorSigmaAlphas[param], scale = priorSigmaBetas[param]))
  else: groupSigma = np.log(norm.pdf(np.log(sigma), priorSigmaAlphas[param], scale = priorSigmaBetas[param]))

  if 'dtnorm' in dist.keys():
    if param in dist['dtnorm']:
      subj = np.sum(np.log(dtnorm(theta, 0, np.inf, loc = phiMu, scale = sigma)), axis = 0)
      groupMu = np.log(dtnorm(phiMu, 0, np.inf, loc = priorMuMus[param], scale = priorMuSigmas[param]))

  if 'dtnorm01' in dist.keys():
    if param in dist['dtnorm01']:
      subj = np.sum(np.log(dtnorm(theta, 0, 1.0, loc = phiMu, scale = sigma)), axis = 0)
      groupMu = np.log(dtnorm(phiMu, 0, 1.0, loc = priorMuMus[param], scale = priorMuSigmas[param]))
  if 'norm' in dist.keys():
    if param in dist['norm']:
      subj = np.sum(np.log(norm.pdf(theta, loc = phiMu, scale = sigma)), axis = 0)
      groupMu = np.log(norm.pdf(phiMu, loc = priorMuMus[param], scale = priorMuSigmas[param]))
  if 'beta' in dist.keys():
    if param in dist['beta']:
      #reparameterization
      a = phiMu * phiSigma
      b = (1 - phiMu) * phiSigma
      subj = np.sum(np.log(beta.pdf(theta, a = a, b = b)), axis = 0)
      #don't allow values that are 1 or 0
      betaex = (phiMu == 1) | (phiMu == 0)
      subj[betaex] = -1 * np.inf
      a = priorMuMus[param] * priorMuSigmas[param]
      b = (1 - priorMuMus[param]) * priorMuSigmas[param]
      groupMu = np.log(beta.pdf(phiMu, a = a, b = b))
      #don't allow values that are 1 or 0
      groupMu[betaex] = -1 * np.inf 
  if 'kde' in dist.keys():
    if param in dist['kde']:
      nChains = len(phiMu)
      groupMu = np.zeros(nChains)
      vals = phiMu
      exclude = (vals < min(kdePriorsMu['support'])) | (vals > max(kdePriorsMu['support']))
      excluded = np.where(exclude)[0]
      groupMu[excluded] = -1 * np.inf #parameters outside the range of the prior are automatically excluded

      if len(excluded) != nChains:
        include = exclude == False
        included = np.where(include)[0]
        includedVals = np.array([findNearest(val, kdePriorsMu['support']) for val in vals[included]]) #find nearest values in the KDE
        includedIdx = np.array([np.where(kdePriorsMu['support'] == val)[0][0] for val in includedVals])
        groupMu[included] = np.log(np.array([kdePriorsMu['density'][idx] for idx in includedIdx]))

      groupSigma = np.zeros(nChains)
      vals = phiSigma
      exclude = (vals < min(kdePriorsSigma['support'])) | (vals > max(kdePriorsSigma['support']))
      excluded = np.where(exclude)[0]
      groupSigma[excluded] = -1 * np.inf #parameters outside the range of the prior are automatically excluded

      if len(excluded) != nChains:
        include = exclude == False
        included = np.where(include)[0]
        includedVals = np.array([findNearest(val, kdePriorsSigma['support']) for val in vals[included]]) #find nearest values in the KDE
        includedIdx = np.array([np.where(kdePriorsSigma['support'] == val)[0][0] for val in includedVals])
        logdens = np.log(np.array([kdePriorsSigma['density'][idx] for idx in includedIdx]))
        groupSigma[included] = np.log(np.array([kdePriorsSigma['density'][idx] for idx in includedIdx]))

  subj[np.isfinite(subj) == False] = -1 * np.inf
  groupMu[np.isfinite(groupMu) == False] = -1 * np.inf
  groupSigma[np.isfinite(groupSigma) == False] = -1 * np.inf
  return subj + groupMu + groupSigma

def log_dens_hyper_and_prior_singleParamBetween(thetas, phiMu, phiSigma, priors, dists, distHiers, distHierAll, param, kdePriorsMu = None, kdePriorsSigma = None):
  
  #construct list indices with the names of all the prior parameters
  priorMuMus, priorMuSigmas, priorSigmaAlphas, priorSigmaBetas = priors

  #if 'precision' in dist.keys(): sigma = 1 / np.sqrt(sigma)
  if 'precision' in dist.keys(): sigma = 1 / phiSigma
  elif 'logSigma' in dist.keys(): sigma = np.exp(phiSigma)
  else: sigma = phiSigma

  if 'logSigma' not in dist.keys(): groupSigma = np.log(gamma.pdf(sigma, priorSigmaAlphas[param], scale = priorSigmaBetas[param]))
  else: groupSigma = np.log(norm.pdf(np.log(sigma), priorSigmaAlphas[param], scale = priorSigmaBetas[param]))

  if 'dtnorm' in dist.keys():
    if param in distHierAll['dtnorm']:
      subj = np.sum(np.log(dtnorm(theta, 0, np.inf, loc = phiMu, scale = sigma)), axis = 0)
      groupMu = np.log(dtnorm(phiMu, 0, np.inf, loc = priorMuMus[param], scale = priorMuSigmas[param]))

  if 'dtnorm01' in dist.keys():
    if param in dist['dtnorm01']:
      subj = np.sum(np.log(dtnorm(theta, 0, 1.0, loc = phiMu, scale = sigma)), axis = 0)
      groupMu = np.log(dtnorm(phiMu, 0, 1.0, loc = priorMuMus[param], scale = priorMuSigmas[param]))
  if 'norm' in dist.keys():
    if param in dist['norm']:
      subj = np.sum(np.log(norm.pdf(theta, loc = phiMu, scale = sigma)), axis = 0)
      groupMu = np.log(norm.pdf(phiMu, loc = priorMuMus[param], scale = priorMuSigmas[param]))
  if 'beta' in dist.keys():
    if param in dist['beta']:
      #reparameterization
      a = phiMu * phiSigma
      b = (1 - phiMu) * phiSigma
      subj = np.sum(np.log(beta.pdf(theta, a = a, b = b)), axis = 0)
      #don't allow values that are 1 or 0
      betaex = (phiMu == 1) | (phiMu == 0)
      subj[betaex] = -1 * np.inf
      a = priorMuMus[param] * priorMuSigmas[param]
      b = (1 - priorMuMus[param]) * priorMuSigmas[param]
      groupMu = np.log(beta.pdf(phiMu, a = a, b = b))
      #don't allow values that are 1 or 0
      groupMu[betaex] = -1 * np.inf 
  if 'kde' in dist.keys():
    if param in dist['kde']:
      nChains = len(phiMu)
      groupMu = np.zeros(nChains)
      vals = phiMu
      exclude = (vals < min(kdePriorsMu['support'])) | (vals > max(kdePriorsMu['support']))
      excluded = np.where(exclude)[0]
      groupMu[excluded] = -1 * np.inf #parameters outside the range of the prior are automatically excluded

      if len(excluded) != nChains:
        include = exclude == False
        included = np.where(include)[0]
        includedVals = np.array([findNearest(val, kdePriorsMu['support']) for val in vals[included]]) #find nearest values in the KDE
        includedIdx = np.array([np.where(kdePriorsMu['support'] == val)[0][0] for val in includedVals])
        groupMu[included] = np.log(np.array([kdePriorsMu['density'][idx] for idx in includedIdx]))

      groupSigma = np.zeros(nChains)
      vals = phiSigma
      exclude = (vals < min(kdePriorsSigma['support'])) | (vals > max(kdePriorsSigma['support']))
      excluded = np.where(exclude)[0]
      groupSigma[excluded] = -1 * np.inf #parameters outside the range of the prior are automatically excluded

      if len(excluded) != nChains:
        include = exclude == False
        included = np.where(include)[0]
        includedVals = np.array([findNearest(val, kdePriorsSigma['support']) for val in vals[included]]) #find nearest values in the KDE
        includedIdx = np.array([np.where(kdePriorsSigma['support'] == val)[0][0] for val in includedVals])
        logdens = np.log(np.array([kdePriorsSigma['density'][idx] for idx in includedIdx]))
        groupSigma[included] = np.log(np.array([kdePriorsSigma['density'][idx] for idx in includedIdx]))


  subj[np.isfinite(subj) == False] = -1 * np.inf
  groupMu[np.isfinite(groupMu) == False] = -1 * np.inf
  groupSigma[np.isfinite(groupSigma) == False] = -1 * np.inf
  return subj + groupMu + groupSigma

def crossover(like, theta, phiMu, phiSigma, data, log_dens_like, params, dist, rp, gammaProp, nJobs, randomGamma = False, recalculate = False):

  nSubj, nChains, nParams = theta.shape[0], theta.shape[1], theta.shape[2]
  #recalculate likelihood in PDA
  if recalculate == True:
    prior = log_dens_hyper(theta, phiMu, phiSigma, dist)
    like = np.ones((nSubj, nChains)) * (-1 * np.inf)
    run = np.isfinite(prior) #indices of non-zero likelihood
    Like = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,run[subj,:]], data[data['subj'] == subject], params) for subj, subject in enumerate(np.unique(data['subj'])))
    for subj in xrange(nSubj): like[subj,run[subj,:]] = Like[subj]

  weight = like + log_dens_hyper(theta, phiMu, phiSigma, dist)

  allVals = [[i for i in xrange(nChains) if i != chain] for chain in xrange(nChains)]
  index = np.array([[np.random.choice(vals, 2, replace = False) for vals in allVals] for subj in xrange(nSubj)])

  #bottom part could be improved: had to resort to a loop, but might work better with a boolean array
  other2 = np.zeros((nSubj,nChains,nParams), dtype = np.float32)
  for subj in xrange(nSubj): other2[subj] = theta[subj,index[subj,:,0]] - theta[subj,index[subj,:,1]]
  new_theta = theta + (gammaProp * other2) + np.random.uniform(-rp, rp, size = (nSubj, nChains, len(rp)))

  #posterior likelihood: likelihood of subject params given group params (log_dens_hyper) and likelihood of data given subject params (log_dens_like)
  prior = log_dens_hyper(new_theta, phiMu, phiSigma, dist)
  new_like = np.ones((nSubj, nChains)) * (-1 * np.inf)
  run = np.isfinite(prior) #indices of non-zero likelihood

  newLike = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(new_theta[subj,run[subj,:]], data[data['subj'] == subject], params) for subj, subject in enumerate(np.unique(data['subj'])))
  for subj in xrange(nSubj): new_like[subj,run[subj,:]] = newLike[subj]
  new_weight = prior + new_like

  #find all chains to be accepted
  accept_idx = np.exp(new_weight - weight) > uniform.rvs(size = (nSubj, nChains))
  accept_idx = accept_idx & np.isfinite(new_weight) #only accept chains with finite likelihood

  theta[accept_idx], like[accept_idx], weight[accept_idx] = new_theta[accept_idx], new_like[accept_idx], new_weight[accept_idx]

  return theta, like, weight

def crossoverBetween(like, thetas, phiMu, phiSigma, nSubjs, data, log_dens_like, ps, dists, distHiers, rp, gammaProps, nJobs, randomGamma = False, recalculate = False):

  nChains, nParams = phiMu.shape
  #recalculate likelihood in PDA, need to rewrite this later!
  if recalculate == True:
    prior = log_dens_hyper(theta, phiMu, phiSigma, dist)
    like = np.ones((nSubj, nChains)) * (-1 * np.inf)
    run = np.isfinite(prior) #indices of non-zero likelihood
    Like = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,run[subj,:]], data[data['subj'] == subject], params) for subj, subject in enumerate(np.unique(data['subj'])))
    for subj in xrange(nSubj): like[subj,run[subj,:]] = Like[subj]

  weight = like + log_dens_hyperBetween(thetas, phiMu, phiSigma, dists, distHiers, len(np.unique(data['subj'])))

  allVals = [[i for i in xrange(nChains) if i != chain] for chain in xrange(nChains)]

  new_thetas = []
  #loop through and create the new_thetas
  for i in xrange(len(thetas)):
    theta, nSubj, gammaProp = thetas[i], nSubjs[i], gammaProps[i]
    nParams = theta.shape[2]
    index = np.array([[np.random.choice(vals, 2, replace = False) for vals in allVals] for subj in xrange(nSubj)]) #randomization is done on each loop

    #bottom part could be improved: had to resort to a loop, but might work better with a boolean array
    other2 = np.zeros((nSubj,nChains,nParams), dtype = np.float32)
    for subj in xrange(nSubj): other2[subj] = theta[subj,index[subj,:,0]] - theta[subj,index[subj,:,1]]
    new_theta = theta + (gammaProp * other2) + np.random.uniform(-rp, rp, size = (nSubj, nChains, len(rp)))
    new_thetas.append(new_theta)

  #posterior likelihood: likelihood of subject params given group params (log_dens_hyper) and likelihood of data given subject params (log_dens_like)
  nSubj = len(np.unique(data['subj']))
  prior = log_dens_hyperBetween(new_thetas, phiMu, phiSigma, dists, distHiers, nSubj)
  new_like = np.ones((nSubj, nChains)) * (-1 * np.inf)
  run = np.isfinite(prior) #indices of non-zero likelihood

  #set up the list of thetas and ps
  new_theta_subjs = []
  for i in xrange(len(new_thetas)): new_theta_subjs += [new_thetas[i][j] for j in xrange(len(new_thetas[i]))]
  
  indices = []
  for i in xrange(len(nSubjs)): indices += [i] * nSubjs[i]
  nSubj = len(np.unique(data['subj']))
  subj_ps = [ps[indices[subj]] for subj in xrange(nSubj)] 

  newLike = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(new_theta_subjs[subj][run[subj,:]], data[data['subj'] == subject], subj_ps[subj]) for subj, subject in enumerate(np.unique(data['subj'])))
  for subj in xrange(nSubj): new_like[subj,run[subj,:]] = newLike[subj]
  new_weight = prior + new_like

  #find all chains to be accepted
  #need to fix this below...
  accept_idx = np.exp(new_weight - weight) > uniform.rvs(size = (nSubj, nChains))
  accept_idx = accept_idx & np.isfinite(new_weight) #only accept chains with finite likelihood

  like[accept_idx], weight[accept_idx] = new_like[accept_idx], new_weight[accept_idx]

  lower = 0
  #loop through the thetas and update them accordingly
  for i in xrange(len(thetas)):
    theta, new_theta = thetas[i], new_thetas[i]
    upper = lower + len(theta)
    accept_theta = accept_idx[lower:upper]
    theta[accept_theta] = new_theta[accept_theta]

    lower = upper

    thetas[i] = theta

  return thetas, like, weight

def crossoverHyper(theta, phiMu, phiSigma, priors, dist, p, rp, kdePriors = None):

  nChains, nParams = phiMu.shape[0], phiMu.shape[1]

  gammaProp = np.random.uniform(.5, 1, size = (nChains,nParams))

  hyperWeight = log_dens_hyper_and_prior(theta, phiMu, phiSigma, priors, dist, p, kdePriors)
  hyperWeight[np.isfinite(hyperWeight) == False] = -1 * np.inf
  allVals = [[i for i in xrange(nChains) if i != chain] for chain in xrange(nChains)]
  index = np.array([[np.random.choice(vals, 2, replace = False) for vals in allVals] for param in xrange(nParams)])
  other2mus, other2sigma = np.zeros(phiMu.shape, dtype = np.float32), np.zeros(phiSigma.shape, dtype = np.float32)
  for param in xrange(nParams):
    other2mus[:,param] = phiMu[index[param,:,0],param] - phiMu[index[param,:,1],param]
    other2sigma[:,param] = phiSigma[index[param,:,0],param] - phiSigma[index[param,:,1],param] 

  new_phiMu, new_phiSigma = np.zeros(phiMu.shape, dtype = np.float32), np.zeros(phiSigma.shape, dtype = np.float32)
  noise = np.random.uniform(-rp, rp, size = (nChains,nParams))
  new_phiMu = phiMu + (gammaProp * other2mus) + noise
  new_phiSigma = phiSigma + (gammaProp * other2sigma) + noise

  new_hyperWeight = log_dens_hyper_and_prior(theta, new_phiMu, new_phiSigma, priors, dist, p, kdePriors)

  new_hyperWeight[np.isnan(new_hyperWeight)] = -1 * np.inf
  accept_idx = np.exp(new_hyperWeight - hyperWeight) > uniform.rvs(size = (nChains, nParams))
  phiMu[accept_idx], phiSigma[accept_idx], hyperWeight[accept_idx] = new_phiMu[accept_idx], new_phiSigma[accept_idx], new_hyperWeight[accept_idx]
  return phiMu, phiSigma, hyperWeight

def crossoverHyperBetween(thetas, phiMu, phiSigma, priors, dists, distHiers, distHierAll, hp, rp, chains, kdePriors = None):

  nChains, nParams = phiMu.shape[0], phiMu.shape[1]

  gammaProp = np.random.uniform(.5, 1, size = (nChains,nParams))

  hyperWeight = log_dens_hyper_and_priorBetween(thetas, phiMu, phiSigma, priors, dists, distHiers, distHierAll, hp, chains, kdePriors)
  hyperWeight[np.isfinite(hyperWeight) == False] = -1 * np.inf
  allVals = [[i for i in xrange(nChains) if i != chain] for chain in xrange(nChains)]
  index = np.array([[np.random.choice(vals, 2, replace = False) for vals in allVals] for param in xrange(nParams)])
  other2mus, other2sigma = np.zeros(phiMu.shape, dtype = np.float32), np.zeros(phiSigma.shape, dtype = np.float32)
  for param in xrange(nParams):
    other2mus[:,param] = phiMu[index[param,:,0],param] - phiMu[index[param,:,1],param]
    other2sigma[:,param] = phiSigma[index[param,:,0],param] - phiSigma[index[param,:,1],param] 

  new_phiMu, new_phiSigma = np.zeros(phiMu.shape, dtype = np.float32), np.zeros(phiSigma.shape, dtype = np.float32)
  noise = np.random.uniform(-rp, rp, size = (nChains,nParams))
  new_phiMu = phiMu + (gammaProp * other2mus) + noise
  new_phiSigma = phiSigma + (gammaProp * other2sigma) + noise

  new_hyperWeight = log_dens_hyper_and_priorBetween(thetas, new_phiMu, new_phiSigma, priors, dists, distHiers, distHierAll, hp, chains, kdePriors)

  new_hyperWeight[np.isnan(new_hyperWeight)] = -1 * np.inf
  accept_idx = np.exp(new_hyperWeight - hyperWeight) > uniform.rvs(size = (nChains, nParams))
  phiMu[accept_idx], phiSigma[accept_idx], hyperWeight[accept_idx] = new_phiMu[accept_idx], new_phiSigma[accept_idx], new_hyperWeight[accept_idx]
  return phiMu, phiSigma, hyperWeight


#this was written for both hierarchical and group level parameters, but could be extended for individual parameters as well
#...this one proposes and updates only the hierarchical parameters. group parameters are updated separately
def crossoverHybrid_subj(gammaProp, like, theta, phiMu, phiSigma, data, log_dens_like, params, hierParams, groupParams, dist, rp, nJobs):

  weight = like + log_dens_hyper(theta[:,:,hierParams], phiMu[:,hierParams], phiSigma[:,hierParams], dist)
  nSubj, nChains, nParams = theta.shape[0], theta.shape[1], theta.shape[2]
  allVals = [[i for i in xrange(nChains) if i != chain] for chain in xrange(nChains)]
  index = np.array([[np.random.choice(vals, 2, replace = False) for vals in allVals] for subj in xrange(nSubj)])

  #bottom part could be improved: had to resort to a loop, but might work better with a boolean array
  new_theta = np.zeros(theta.shape)
  other2 = np.zeros((nSubj,nChains,nParams), dtype = np.float32)
  for subj in xrange(nSubj): other2[subj] = theta[subj,index[subj,:,0]] - theta[subj,index[subj,:,1]]
  new_theta = theta + (gammaProp * other2) + np.random.uniform(-rp, rp, size = (nSubj, nChains, 1))
  new_theta[:,:,groupParams] = theta[:,:,groupParams] #have to include group parameters as well

  #posterior likelihood: likelihood of subject params given group params (log_dens_hyper) and likelihood of data given subject params (log_dens_like)
  prior = log_dens_hyper(new_theta[:,:,hierParams], phiMu[:,hierParams], phiSigma[:,hierParams], dist)
  new_like = np.ones((nSubj, nChains)) * (-1 * np.inf)
  run = np.isfinite(prior) #indices of non-zero likelihood
  newLike = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(new_theta[subj,run[subj,:]], data[data['subj'] == subject], params) for subj, subject in enumerate(np.unique(data['subj'])))
  for subj in xrange(nSubj): new_like[subj,run[subj,:]] = newLike[subj]
  new_weight = prior + new_like

  #find all chains to be accepted
  accept_idx = np.exp(new_weight - weight) > uniform.rvs(size = (nSubj, nChains))
  accept_idx = accept_idx & np.isfinite(new_weight)

  theta[accept_idx], like[accept_idx], weight[accept_idx] = new_theta[accept_idx], new_like[accept_idx], new_weight[accept_idx]

  return theta, like, weight

def crossoverHybrid_group(gammaProp, like, theta, phiMu, data, log_dens_like, params, groupParams, hierParams, dist, priors, rp, nJobs):

  nSubj, nChains, nParams = theta.shape[0], theta.shape[1], theta.shape[2]

  weight = np.sum(like, axis = 0) + log_dens_prior(phiMu, priors, dist)

  #group level parameters: these come from phiMu rather than theta
  new_phiMu, new_theta = np.zeros(phiMu.shape), np.zeros(theta.shape)
  allVals = [[i for i in xrange(nChains) if i != chain] for chain in xrange(nChains)]
  index = np.array([np.random.choice(vals, 2, replace = False) for vals in allVals])
  other2 = phiMu[index[:,0]] - phiMu[index[:,1]]

  new_phiMu = phiMu + (gammaProp * other2) + np.random.uniform(-rp, rp, size = (nChains, 1)) 
  new_theta[:,:,groupParams] = new_phiMu.reshape(1,nChains,len(groupParams)) + 0.0
  new_theta[:,:,hierParams] = theta[:,:,hierParams]

  prior = log_dens_prior(new_phiMu, priors, dist)
  new_like = np.ones((nSubj, nChains)) * (-1 * np.inf) #keep this varied by subject so you can reuse it
  run = np.isfinite(prior) #indices of non-zero likelihood
  newLike = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(new_theta[subj,run], data[data['subj'] == subject], params) for subj, subject in enumerate(np.unique(data['subj'])))
  for subj in xrange(nSubj): new_like[subj,run] = newLike[subj]
  new_weight = prior + np.sum(new_like, axis = 0)

  #find all chains to be accepted
  accept_idx = np.exp(new_weight - weight) > uniform.rvs(size = nChains)

  phiMu[accept_idx], like[:,accept_idx], weight[accept_idx] = new_phiMu[accept_idx], new_like[:,accept_idx], new_weight[accept_idx]
  theta[:,:,groupParams] = phiMu.reshape((1, nChains, len(groupParams))) + 0.0

  return phiMu, theta, weight, like

def crossoverGroup(like, weight, theta, priors, data, log_dens_like, params, dist, rp):

  nChains, nParams = theta.shape[0], theta.shape[1]
  gammaProp = 2.38/np.sqrt(2 * nParams)
  allVals = [[i for i in xrange(nChains) if i != chain] for chain in xrange(nChains)]
  index = np.array([np.random.choice(vals, 2, replace = False) for vals in allVals])
  other2 = theta[index[:,0]] - theta[index[:,1]]

  new_theta = theta + (gammaProp * other2) + np.random.uniform(-rp, rp, size = (nChains, 1))

  #posterior likelihood: likelihood of subject params given group params (log_dens_hyper) and likelihood of data given subject params (log_dens_like)
  prior = log_dens_prior(theta, priors, dist)

  new_like = np.ones(nChains) * (-1 * np.inf)
  run = np.isfinite(prior) #indices of non-zero likelihood
  nSubj = len(np.unique(data['subj']))

  subjLike = np.zeros((nSubj,np.sum(run)))
  newLike = Parallel(n_jobs = -1)(delayed(log_dens_like)(new_theta[run], data[data['subj'] == subj], params) for subj in xrange(nSubj))
  for subj in xrange(nSubj): subjLike[subj] = newLike[subj]
  new_like[run] = np.sum(subjLike, axis = 0)
    
  new_weight = prior + new_like

  #find all chains to be accepted
  accept_idx = np.exp(new_weight - weight) > uniform.rvs(size = nChains)
  accept_idx = accept_idx & (np.isfinite(new_weight)) #only accept chains with finite likelihood

  theta[accept_idx], like[accept_idx], weight[accept_idx] = new_theta[accept_idx], new_like[accept_idx], new_weight[accept_idx]
  return theta, like, weight

#need to modify this function so it can take different prior distributions
def crossoverSubject(gammaProp, like, theta, priors, data, log_dens_like, params, dist, rp, nChainBlocks = 4, nJobs = -1):

  weight = like + log_dens_prior(theta, priors, dist)

  nChains, nParams = theta.shape[0], theta.shape[1]
  allVals = [[i for i in xrange(nChains) if i != chain] for chain in xrange(nChains)]
  index = np.array([np.random.choice(vals, 2, replace = False) for vals in allVals])
  other2 = theta[index[:,0]] - theta[index[:,1]]

  new_theta = theta + (gammaProp * other2) + np.random.uniform(-rp, rp, size = (nChains, 1))

  prior = log_dens_prior(new_theta, priors, dist)

  new_like = np.ones(nChains) * (-1 * np.inf)
  run = np.isfinite(prior) #indices of non-zero likelihood

  cB = nChains / nChainBlocks
  new_like = np.ones(nChains) * (-1 * np.inf)
  #slicing below is tricky... take the chain block in new_theta, then select the appropriate chain block in run
  newLike = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(new_theta[cB*c:cB*(c+1)][run[cB*c:cB*(c+1)]], data, params) for c in xrange(nChainBlocks))
  for c in xrange(nChainBlocks): new_like[cB*c:cB*(c+1)][run[cB*c:cB*(c+1)]] = newLike[c]
  new_weight = prior + new_like

  #find all chains to be accepted
  accept_idx = np.exp(new_weight - weight) > uniform.rvs(size = nChains)

  theta[accept_idx], like[accept_idx], weight[accept_idx] = new_theta[accept_idx], new_like[accept_idx], new_weight[accept_idx]
  return theta, like, weight

#this one is used for multiple subjects
def crossoverSubjects(gammaProp, like, theta, priors, data, log_dens_like, params, dist, rp, nChainBlocks = 4, nJobs = -1):

  weight = like + log_dens_priorSubjects(theta, priors, dist)
  nSubj, nChains, nParams = theta.shape[0], theta.shape[1], theta.shape[2]

  allVals = [[i for i in xrange(nChains) if i != chain] for chain in xrange(nChains)]
  index = np.array([[np.random.choice(vals, 2, replace = False) for vals in allVals] for subj in xrange(nSubj)])

  #bottom part could be improved: had to resort to a loop, but might work better with a boolean array
  other2 = np.zeros((nSubj,nChains,nParams), dtype = np.float32)
  for subj in xrange(nSubj): other2[subj] = theta[subj,index[subj,:,0]] - theta[subj,index[subj,:,1]]
  new_theta = theta + (gammaProp * other2) + np.random.uniform(-rp, rp, size = (nSubj, nChains,1))

  prior = log_dens_priorSubjects(new_theta, priors, dist)
  new_like = np.ones((nSubj, nChains)) * (-1 * np.inf)
  run = np.isfinite(prior) #indices of non-zero likelihood

  newLike = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(new_theta[subj,run[subj,:]], data[data['subj'] == subject], params) for subj, subject in enumerate(np.unique(data['subj'])))
  for subj in xrange(nSubj): new_like[subj,run[subj,:]] = newLike[subj]
  new_weight = prior + new_like

  #find all chains to be accepted
  accept_idx = np.exp(new_weight - weight) > uniform.rvs(size = (nSubj, nChains))
  accept_idx = accept_idx & np.isfinite(new_weight) #only accept chains with finite likelihood

  theta[accept_idx], like[accept_idx], weight[accept_idx] = new_theta[accept_idx], new_like[accept_idx], new_weight[accept_idx]
  return theta, like, weight

def migration(theta, phiMu, phiSigma, like, data, log_dens_like, params, dist, rp):
  nChains, nParams = theta.shape[0], theta.shape[1]
  eta = max(random.choice(range(1, nChains + 1)), 2) #eta: number of chains to be used: at least 2
  k = range(nChains)
  random.shuffle(k)
  chains = k[:eta]
  rotated = chains[1:] + [chains[0]]

  weight = like + log_dens_hyper(theta.reshape(1,nChains,nParams), phiMu, phiSigma, dist)
  weight = weight.reshape(nChains)
  new_like = np.ones(eta) * (-1 * np.inf)
  new_theta = (theta[rotated] + np.random.uniform(-rp,rp,size = (eta,1))).reshape(1,len(rotated),nParams) #have to reshape: log_dens_hyper requires subject index first
  priorDens = log_dens_hyper(new_theta, phiMu[chains], phiSigma[chains], dist) #evaluate new chains (rotated) under old ones (chains)
  priorDens = priorDens.reshape(len(rotated))
  run = np.isfinite(priorDens)

  new_theta = new_theta.reshape(len(rotated), nParams)
  if np.sum(run) > 0: new_like[run] = log_dens_like(new_theta[run], data, params)
  new_weight = new_like + priorDens

  accept_idx = np.exp(new_weight - weight[chains]) > uniform.rvs(size = eta) #accept_idx is as big as len(chains) and len(rotated)
  accept_idx = accept_idx & (np.isfinite(new_weight))
  chains = np.array(chains)

  theta[chains[accept_idx]] = new_theta[accept_idx]
  like[chains[accept_idx]] = new_like[accept_idx]
  weight[chains[accept_idx]] = new_weight[accept_idx]

  return theta, like, weight

def migrationSubject(theta, like, data, log_dens_like, params, dist, priors, rp, nJobs = -1, parallel = False):
  nChains, nParams = theta.shape
  eta = max(random.choice(range(1, nChains + 1)), 2) #eta: number of chains to be used: at least 2
  k = range(nChains)
  random.shuffle(k)
  chains = k[:eta]
  rotated = chains[1:] + [chains[0]]

  weight = like + log_dens_prior(theta, priors, dist)
  weight = weight.reshape(nChains)
  new_like = np.ones(eta) * (-1 * np.inf)
  new_theta = (theta[rotated] + np.random.uniform(-rp,rp,size = (eta,1)))
  priorDens = log_dens_prior(new_theta, priors, dist) #evaluate new chains (rotated) under old ones (chains)
  run = np.isfinite(priorDens)

  new_theta = new_theta.reshape(len(rotated), nParams)
  new_like = np.zeros(len(chains))
  toRun = np.where(run)[0]

  if parallel == True:
    newLike = np.array(Parallel(n_jobs = nJobs)(delayed(log_dens_like)(new_theta[chain], data, params) for chain in toRun)).flatten()
    for c, chain in enumerate(xrange(len(toRun))): new_like[chain] = newLike[c] 
  else: new_like[run] = log_dens_like(new_theta[run], data, params)

  new_weight = new_like + priorDens

  accept_idx = np.exp(new_weight - weight[chains]) > uniform.rvs(size = eta) #accept_idx is as big as len(chains) and len(rotated)
  chains = np.array(chains)

  theta[chains[accept_idx]] = new_theta[accept_idx]
  like[chains[accept_idx]] = new_like[accept_idx]
  weight[chains[accept_idx]] = new_weight[accept_idx]

  return theta, like, weight

def migrationHybrid_subj(theta, phiMu, phiSigma, like, data, log_dens_like, params, hierParams, groupParams, dist, rp):
  nChains, nParams = theta.shape[0], theta.shape[1]
  eta = max(random.choice(range(1, nChains + 1)), 2) #eta: number of chains to be used: at least 2
  k = range(nChains)
  random.shuffle(k)
  chains = k[:eta]
  rotated = chains[1:] + [chains[0]]

  weight = like + log_dens_hyper(theta.reshape(1,nChains,nParams)[:,:,hierParams], phiMu, phiSigma, dist).reshape(nChains) #need to index only the hierParams in theta
  new_like = np.ones(eta) * (-1 * np.inf)
  new_theta = (theta[rotated] + np.random.uniform(-rp,rp,size = (eta,1))).reshape(len(rotated),nParams)
  for param in groupParams: new_theta[:,param] = theta[rotated,param]
  new_theta = new_theta.reshape(1,len(rotated),len(params))

  priorDens = log_dens_hyper(new_theta[:,:,hierParams], phiMu[chains], phiSigma[chains], dist) #evaluate new chains (rotated) under old ones (chains)
  priorDens = priorDens.reshape(len(rotated))
  run = np.isfinite(priorDens)

  new_theta = new_theta.reshape(len(rotated),len(params))
  if np.sum(run) > 0: new_like[run] = log_dens_like(new_theta[run], data, params)
  new_weight = new_like + priorDens

  accept_idx = np.exp(new_weight - weight[chains]) > uniform.rvs(size = eta) #accept_idx is as big as len(chains) and len(rotated)
  accept_idx = accept_idx & (np.isfinite(new_weight))

  chains = np.array(chains)

  theta[chains[accept_idx]] = new_theta[accept_idx]
  like[chains[accept_idx]] = new_like[accept_idx]
  weight[chains[accept_idx]] = new_weight[accept_idx]

  return theta, like, weight

def migrationHybrid_group(theta, phiMu, like, data, log_dens_like, priors, params, groupParams, hierParams, dist, rp, nJobs):

  nSubj, nChains = theta.shape[0], theta.shape[1]
  eta = max(random.choice(range(1, nChains + 1)), 2) #eta: number of chains to be used: at least 2
  k = range(nChains)
  random.shuffle(k)
  chains = k[:eta]
  rotated = chains[1:] + [chains[0]]

  weight = np.sum(like, axis = 0) + log_dens_prior(phiMu, priors, dist)

  noise = np.random.uniform(-rp, rp, size = (eta,1))
  new_phiMu = np.zeros((len(rotated), len(groupParams)))
  new_phiMu = phiMu[rotated] + noise
  new_theta = theta[:,chains] + 0.0
  new_theta[:,:,groupParams] = new_phiMu.reshape(1,len(rotated),len(groupParams)) + 0.0

  prior = log_dens_prior(new_phiMu, priors, dist)
  new_like = np.ones((nSubj, len(rotated))) * (-1 * np.inf) #keep this varied by subject so you can reuse it
  run = np.isfinite(prior) #indices of non-zero likelihood
  newLike = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(new_theta[subj,run], data[data['subj'] == subject], params) for subj, subject in enumerate(np.unique(data['subj'])))
  for subj in xrange(nSubj): new_like[subj,run] = newLike[subj]
  new_weight = prior + np.sum(new_like, axis = 0)

  accept_idx = np.exp(new_weight - weight[chains]) > uniform.rvs(size = eta)
  accept_idx = accept_idx & (np.isfinite(new_weight))
  chains = np.array(chains)
  phiMu[chains[accept_idx]], weight[chains[accept_idx]], like[:,chains[accept_idx]] = new_phiMu[accept_idx], new_weight[accept_idx], new_like[:,accept_idx]
  #NOTE BELOW: I wanted to do this with two lists: theta[:,chain[accept_idx],groupParams], but Python won't let you index with two lists. Might need to find another solution
  for i, param in enumerate(groupParams):
    theta[:,chains[accept_idx],param] = phiMu[chains[accept_idx],i].reshape(1,len(chains[accept_idx]))

  return phiMu, theta, weight, like

#due to only operating on a single parameter, prior mu and sigma are separate
#theta and the phis are only of a single parameter
#...param: this is an index, not a parameter name
def migrationHyper(theta, phiMu, phiSigma, priors, dist, param, p, rp, kdePriors = None):

  if kdePriors != None:
    if p[param] in kdePriors['mu'].keys():
      kdePriorsMu, kdePriorsSigma = kdePriors['mu'][p[param]], kdePriors['sigma'][p[param]]
    else: kdePriorsMu, kdePriorsSigma = None, None
  else: kdePriorsMu, kdePriorsSigma = None, None

  nChains = len(phiMu)
  eta = max(random.choice(range(1, nChains + 1)), 2) #eta: number of chains to be used: at least 2
  k = range(nChains)
  random.shuffle(k)
  chains = k[:eta]
  rotated = chains[1:] + [chains[0]]
 
  noise = np.random.uniform(-rp, rp, size = eta)
  new_phiMu, new_phiSigma = phiMu[rotated] + noise, phiSigma[rotated] + noise
  hyperWeight = log_dens_hyper_and_prior_singleParam(theta, phiMu, phiSigma, priors, dist, param, kdePriorsMu, kdePriorsSigma)
  new_hyperWeight = log_dens_hyper_and_prior_singleParam(theta[:,chains], new_phiMu, new_phiSigma, priors, dist, param, kdePriorsMu, kdePriorsSigma)
  
  new_hyperWeight[np.isnan(new_hyperWeight)] = -1 * np.inf
  accept_idx = np.exp(new_hyperWeight - hyperWeight[chains]) > uniform.rvs(size = eta) #accept_idx is eta by nParams
  chains = np.array(chains)
  
  phiMu[chains[accept_idx]] = new_phiMu[accept_idx]
  phiSigma[chains[accept_idx]] = new_phiSigma[accept_idx]
  hyperWeight[chains[accept_idx]] = new_hyperWeight[accept_idx]

  return phiMu, phiSigma, hyperWeight

#this differs from the function above by looping through each of the parameters within the function, rather than in samplingHier
def migrationHyperBetween(thetas, phiMu, phiSigma, priors, dist, ps, hp, rp, kdePriors = None):

  if kdePriors != None:
    if p[param] in kdePriors['mu'].keys():
      kdePriorsMu, kdePriorsSigma = kdePriors['mu'][p[param]], kdePriors['sigma'][p[param]]
    else: kdePriorsMu, kdePriorsSigma = None, None
  else: kdePriorsMu, kdePriorsSigma = None, None

  nChains, nParams = phiMu.shape
  hyperWeight = np.zeros((nChains, nParams))
  for param in hp:
    
    eta = max(random.choice(range(1, nChains + 1)), 2) #eta: number of chains to be used: at least 2
    k = range(nChains)
    random.shuffle(k)
    chains = k[:eta]
    rotated = chains[1:] + [chains[0]]

    noise = np.random.uniform(-rp, rp, size = eta)
    new_phiMu, new_phiSigma = phiMu[rotated,hp.index(param)] + noise, phiSigma[rotated,hp.index(param)] + noise
    
    #create an array that contains all of the theta values that match the parameter
    ths = []
    for g in xrange(len(thetas)):
      theta, p = thetas[g], ps[g]
 
      if param in p:
        th = theta[:,:,p.index(param)]
        ths.append(th)
    for g2 in xrange(len(ths)):
      if g2 == 0: tht = ths[0]
      else: tht = np.vstack((tht, ths[g2]))

    hyperWeight[:, hp.index(param)] = log_dens_hyper_and_prior_singleParam(tht, phiMu[:,hp.index(param)], phiSigma[:,hp.index(param)], priors, dist, hp.index(param), kdePriorsMu, kdePriorsSigma)
    new_hyperWeight = log_dens_hyper_and_prior_singleParam(tht[:,chains], new_phiMu, new_phiSigma, priors, dist, hp.index(param), kdePriorsMu, kdePriorsSigma)
  
    new_hyperWeight[np.isnan(new_hyperWeight)] = -1 * np.inf
    accept_idx = np.exp(new_hyperWeight - hyperWeight[chains,hp.index(param)]) > uniform.rvs(size = eta) #accept_idx is eta by nParams
    chains = np.array(chains)
  
    phiMu[chains[accept_idx], hp.index(param)] = new_phiMu[accept_idx]
    phiSigma[chains[accept_idx], hp.index(param)] = new_phiSigma[accept_idx]
    hyperWeight[chains[accept_idx], hp.index(param)] = new_hyperWeight[accept_idx]

  return phiMu, phiSigma, hyperWeight

##### SAMPLING PHASE #####

def phiSample(phiMu, phiSigma, nChains):
  nSamples, nParams = phiMu.shape
  vals = range(nSamples)
  shuffle(vals)
  return phiMu[vals[0:nChains]], phiSigma[vals[0:nChains]]

def samplingHier(data, p, dist, log_dens_like, starts, priors, nChains = 4, nmc = 100, burnin = 0, thin = 1, nJobs = 4, informedStart = -1, cont = False, recalculate = False, pb = False, **kwargs):

  print 'HIERARCHICAL FIT'
  print '----------------'
  print 'nChains: ' + str(nChains) + ' Burnin: ' + str(burnin) + ' nmc: ' + str(nmc) + ' thin: ' + str(thin)
  print '# parameters: ' + str(len(p))
  print p
  print 'dist: ' + str(dist)
  
  if 'gamma1' in kwargs.keys(): gamma1 = kwargs['gamma1']
  else: gamma1 = False
  if 'gammat0' in kwargs.keys(): gammat0 = kwargs['gammat0']
  else: gammat0 = False  
  
  if gamma1 == True: print 'Using gamma = 1 every 10th iteration'
  elif gammat0 == True: print 'Using gamma = 1 for t0/st0 for every 10th iteration'

  #gammaUnif: gamma = uniform(.5, 1)
  if 'gammaUnif' in kwargs.keys():
    gammaUnif = kwargs['gammaUnif']
    if gammaUnif == True: print 'Gamma sampled from Uniform(.5, 1.0)'
  else: gammaUnif = False

  #message about recalculating likelihoods
  if recalculate != False: print 'Recalculating likelihoods every ', recalculate, ' samples. Welcome to PDA Land!!!'

  if burnin == 0: print '***NO BURN-IN***'

  #option of changing the perturbation parameter
  if 'rp' in kwargs.keys(): rp = kwargs['rp']
  else: rp = .001
  
  if 'minSigma' in kwargs.keys(): minSigma = kwargs['minSigma']
  else: minSigma = 0.0

  nParams, nSubj = len(p), len(np.unique(data['subj']))
  use_theta, use_phiMu, use_phiSigma = np.zeros((nSubj, nChains, nParams), dtype = np.float32), np.zeros((nChains, nParams), dtype = np.float32), np.zeros((nChains, nParams), dtype = np.float32)

  #if there are KDE priors, designate them now
  if 'kdePriors' in kwargs.keys():
    kdePriors = kwargs['kdePriors']
  else:
    kdePriors = None

  priorMuMus = np.array([priors[param + '_mumu'] for param in p])
  priorMuSigmas = np.array([priors[param + '_musigma'] for param in p])
  priorSigmaAlphas = np.array([priors[param + '_sigmaalpha'] for param in p])
  priorSigmaBetas = np.array([priors[param + '_sigmabeta'] for param in p])
  priors = [priorMuMus, priorMuSigmas, priorSigmaAlphas, priorSigmaBetas]

  ##### START VALUES #####

  ###randomly generated start values
  #...these scale factors denote how sigma values are scaled relative to the means
  sigmaScale = 8.5
  phiSigmaScale, sigmaSigmaScale = 8.5, 16.0

  if informedStart == -1:
    print 'Creating values...'
    for param in p:
      if type(starts[param]) == tuple:
        if len(starts[param]) == 2: #tuple of length 2: start values for mu and sigma
          start = starts[param][0]
          sigma = starts[param][1]
          phiSigmaStart = starts[param][1]
          sigmaSigma = phiSigmaStart / sigmaSigmaScale
      else:
        start = starts[param]
        if starts[param] != 0.0: sigma, phiSigmaStart, sigmaSigma = np.abs(float(starts[param]) / sigmaScale), np.abs(float(starts[param]) / phiSigmaScale), np.abs(float(starts[param]) / sigmaSigmaScale)
        elif starts[param] == 0.0: sigma, phiSigmaStart, sigmaSigma = .05, .025, .05

      use_phiSigma[:,p.index(param)] = rtnorm(minSigma, np.inf, phiSigmaStart, sigmaSigma, size = nChains)

      if 'dtnorm' in dist.keys():
        if p.index(param) in dist['dtnorm']:
          use_phiMu[:,p.index(param)] = rtnorm(0, np.inf, start, sigma, size = nChains)
          use_theta[:,:,p.index(param)] = rtnorm(0, np.inf, use_phiMu[:,p.index(param)].reshape(1,nChains), use_phiSigma[:,p.index(param)].reshape(1,nChains) / phiSigmaScale, size = (nSubj, nChains))
      if 'dtnorm01' in dist.keys():
        if p.index(param) in dist['dtnorm01']:
          use_phiMu[:,p.index(param)] = rtnorm(0.0, 1.0, start, sigma, size = nChains)
          use_theta[:,:,p.index(param)] = rtnorm(0.0, 1.0, use_phiMu[:,p.index(param)], use_phiSigma[:,p.index(param)] / phiSigmaScale, size = (nSubj,nChains))
      if 'dtnorm01beta' in dist.keys():
        if p.index(param) in dist['dtnorm01beta']:
          use_phiMu[:,p.index(param)] = rtnorm(0.0, 1.0, start, sigma, size = nChains)
          use_theta[:,:,p.index(param)] = rtnorm(0.0, 1.0, use_phiMu[:,p.index(param)], use_phiSigma[:,p.index(param)] / phiSigmaScale, size = (nSubj,nChains))
      if 'beta' in dist.keys():
        if p.index(param) in dist['beta']:
          use_phiMu[:,p.index(param)] = rtnorm(0.0, 1.0, start, sigma, size = nChains)
          use_theta[:,:,p.index(param)] = rtnorm(0.0, 1.0, use_phiMu[:,p.index(param)], use_phiSigma[:,p.index(param)] / phiSigmaScale, size = (nSubj,nChains))
          #phiSigmaStart = np.abs(float(starts[param])) #beta does not use sigma, but sample size - this tends to be much bigger than the sigma
          #use_phiSigma[:,p.index(param)] = rtnorm(minSigma, np.inf, phiSigmaStart, sigmaSigma, size = nChains)
      if 'dtnormsz' in dist.keys():
        if p.index(param) in dist['dtnormsz']:
          use_phiMu[:,p.index(param)] = rtnorm(0.0, szMax, start, sigma, size = nChains)
          use_theta[:,:,p.index(param)] = rtnorm(0.0, szMax, use_phiMu[:,p.index(param)], use_phiSigma[:,p.index(param)] / phiSigmaScale, size = (nSubj,nChains))
      if 'norm' in dist.keys():
        if p.index(param) in dist['norm']: 
          use_phiMu[:,p.index(param)] = norm.rvs(start, sigma, size = nChains)
          use_theta[:,:,p.index(param)] = norm.rvs(use_phiMu[:,p.index(param)].reshape(1,nChains), use_phiSigma[:,p.index(param)].reshape(1,nChains) / phiSigmaScale, size = (nSubj, nChains))
      if 'kde' in dist.keys():
        if p.index(param) in dist['kde']:
          #sample randomly from the KDE for start points
          probMu = kdePriors['mu'][param]['density'] / np.sum(kdePriors['mu'][param]['density'])
          probSigma = kdePriors['sigma'][param]['density'] / np.sum(kdePriors['sigma'][param]['density'])
          use_phiMu[:,p.index(param)] = [np.random.choice(kdePriors['mu'][param]['support'], p = probMu) for chain in xrange(nChains)]
          use_phiSigma[:,p.index(param)] = [np.random.choice(kdePriors['sigma'][param]['support'], p = probSigma) for chain in xrange(nChains)]
          use_theta[:,:,p.index(param)] = [np.random.choice(kdePriors['mu'][param]['support'], p = probMu) for chain in xrange(nChains)]

    if 'precision' in dist.keys():
      print 'Using precision scale...'
      #use_phiSigma = 1 / use_phiSigma**2
      use_phiSigma = 1 / use_phiSigma
    if 'logSigma' in dist.keys():
      print 'Sampling log(sigma)...'
      use_phiSigma = np.exp(use_phiSigma)
          
  #use previous fits
  elif informedStart != -1:
    if cont == False:
      print 'Loading start values from a previous fit...'
      use_phiMu = getPhiMu(informedStart)[0:nChains,-1]
      use_phiSigma = getPhiSigma(informedStart)[0:nChains,-1]
      for subj in xrange(nSubj): use_theta[subj,:] = getSubj(subj,informedStart)[0:nChains,-1]
    elif cont == True:
      print 'Continuing a previous fit...'
      theta0 = getSubj(0, informedStart)[0:nChains]
      nChains, nPrevious, nParams = theta0.shape
      theta = np.zeros((nSubj,nChains,nPrevious,nParams))
      theta[0] = theta0
      for subj in xrange(1,nSubj): theta[subj] = getSubj(subj,informedStart)[0:nChains]

      phiMu, phiSigma = getPhiMu(informedStart)[0:nChains], getPhiSigma(informedStart)[0:nChains]
      use_theta, use_phiMu, use_phiSigma = theta[:,:,nPrevious-1], phiMu[:,nPrevious-1], phiSigma[:,nPrevious-1]

      theta = np.concatenate((theta, np.zeros((nSubj,nChains,nmc,nParams))), axis = 2)
      phiMu = np.concatenate((phiMu, np.zeros((nChains,nmc,nParams))), axis = 1)
      phiSigma = np.concatenate((phiSigma, np.zeros((nChains,nmc,nParams))), axis = 1)
      like = np.concatenate((getLike(informedStart)[:,0:nChains], np.zeros((nSubj,nChains,nmc))), axis = 2)
      weight = np.concatenate((getWeight(informedStart)[:,0:nChains], np.zeros((nSubj,nChains,nmc))), axis = 2)
      hyperWeight = np.concatenate((getHyperWeight(informedStart)[0:nChains], np.zeros((nChains,nmc,nParams))), axis = 1)
      
      use_weight, useHyperWeight = weight[:,:,nPrevious-1], hyperWeight[:,nPrevious-1]

      del theta0
      gc.collect()

  print 'Start values generated...'

  ### generate the initial set of likelihoods
  use_like = np.array(Parallel(n_jobs = nJobs)(delayed(log_dens_like)(use_theta[subj], data[data['subj'] == subject], p) for subj, subject in enumerate(np.unique(data['subj']))))
  prior = log_dens_hyper(use_theta, use_phiMu, use_phiSigma, dist)
  use_weight = prior + use_like

  ### parameter check ###
  use_hyperWeight = log_dens_hyper_and_prior(use_theta, use_phiMu, use_phiSigma, priors, dist, p, kdePriors)
  if (np.mean(np.isfinite(np.mean(use_weight))) == 1.0) & (np.mean(np.isfinite(use_hyperWeight)) == 1.0):
    print 'Check passed!'

  else:
    print 'Check failed...'
    if np.mean(np.isfinite(np.mean(use_like))) != 1.0:
      print 'Bad likelihood values...'
      print np.mean(np.isfinite(use_like))
    if np.mean(np.isfinite(prior)) != 1.0:
      print 'Subject parameters illegal under hyperparameters... ', np.mean(np.isfinite(prior))
      priorDebug = log_dens_hyper_separate(use_theta, use_phiMu, use_phiSigma, dist) #returns as nSubjects x nChains x nParams
      for param in p:
        if np.mean(np.isfinite(priorDebug[:,:,p.index(param)])) != 1.0:
          print param
      #print use_theta[:,:,p.index('crit')]
      #print use_phiSigma[:,p.index('crit')]

    if (np.mean(np.isfinite(use_hyperWeight)) != 1.0):
      print 'Hyperparameters illegal under priors...'     
      for param in p:
        if np.mean(np.isfinite(use_hyperWeight[:,p.index(param)])) != 1.0:
          print param
      print np.mean(np.isfinite(use_hyperWeight))


  #replace nans with negative inf
  use_weight[np.isnan(use_weight)] = -1 * np.inf
  use_like[np.isnan(use_like)] = -1 * np.inf
 
  print 'Beginning sampling...'
  if 'migrationStart' in kwargs.keys(): migrationStart = kwargs['migrationStart']
  else: migrationStart = 15000
  if 'migrationStop' in kwargs.keys(): migrationStop = kwargs['migrationStop']
  else: migrationStop = 10**1000
  gc.collect()

  #create structures right away if burnin is zero
  if (burnin == 0) & (cont == False):
    theta, phiMu, phiSigma = np.zeros((nSubj, nChains, nmc, nParams), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32)
    hyperWeight, like, weight = np.zeros((nChains, nmc, nParams), dtype = np.float32), np.zeros((nSubj, nChains, nmc), dtype = np.float32), np.zeros((nSubj, nChains, nmc), dtype = np.float32)  
    phiMu[:,0], phiSigma[:,0], hyperWeight[:,0], theta[:,:,0], like[:,:,0], weight[:,:,0] = use_phiMu, use_phiSigma, use_hyperWeight, use_theta, use_like, use_weight
    i = 0
     

  ### MCMC loop ###
  MCMC = True
  step = 0
  status = ''
  
  if cont == True:
    step = nPrevious
    i = nPrevious #ensure that MCMC starts where it left off
    nmc += nPrevious
  if step < burnin: print 'Burn-in period!'

  if 'randomGamma' in kwargs.keys(): randomGamma = kwargs['randomGamma']
  else: randomGamma = False 

  if type(rp) == np.float: rp = np.array([rp])
  rp1 = rp

  while MCMC == True:

    step += 1

    #recalculate likelihoods (for the PDA route)
    recalc = False
    if recalculate != False:
      #recalculate less frequently during burn-in
      if step < burnin:
        if step % (recalculate * 2) == 0: recalc = True
      else:
        if step % recalculate == 0: recalc = True

    #gamma parameters
    if ((gamma1 == True) | (step < burnin)) & ((step + 5) % 10 == 0):
      gammaProp = .98
      if recalculate == True: recalc = True
    elif (gammat0 == True) & (step % 10 == 0):
      t = [p.index(param) for param in p if ('t0' in param) | ('st0' in param)] #just upscale the problem parameters
      if gammaUnif == False: gammaProp = np.ones((1,1,nParams)) * (2.38 / np.sqrt(2 * nParams))
      elif gammaUnif == True: gammaProp = np.ones((nSubj,nChains,nParams)) * np.random.uniform(.5, 1.0, size = (nSubj,nChains,1))
      gammaProp[:,:,t] = 1.1
    elif gammaUnif == True: gammaProp = gammaProp = np.ones((nSubj,nChains,nParams)) * np.random.uniform(.5, 1.0, size = (nSubj,nChains,1))
    elif gammaUnif == False:  gammaProp = 2.38 / np.sqrt(2 * nParams)
    #gammaProp = np.random.uniform(.5, 1)
    if (step < burnin) | (step % 15 == 0): rp = rp1 * 5
    else: rp = rp1

    ### create the data structures if burnin period is over
    if step == burnin:
      status = 'Burnin over'
      theta, phiMu, phiSigma = np.zeros((nSubj, nChains, nmc, nParams), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32)
      hyperWeight, like, weight = np.zeros((nChains, nmc, nParams), dtype = np.float32), np.zeros((nSubj, nChains, nmc), dtype = np.float32), np.zeros((nSubj, nChains, nmc))   
      i = 0

    ### crossover hyper step: sample new group parameters ###
    chains = range(nChains)
    shuffle(chains)
    
    use_phiMu, use_phiSigma, use_hyperWeight = crossoverHyper(use_theta[:,chains], use_phiMu, use_phiSigma, priors, dist, p, rp, kdePriors)

    ### crossover step: sample new subject parameters ###
    chains = range(nChains)
    shuffle(chains)
 
    use_theta, use_like, use_weight = crossover(use_like, use_theta, use_phiMu[chains], use_phiSigma[chains], data, log_dens_like, p, dist, rp, gammaProp, nJobs, randomGamma = randomGamma, recalculate = recalc)
    #use_theta, use_like, use_weight = crossover(use_like, use_theta, use_phiMu, use_phiSigma, data, log_dens_like, p, dist, rp, gammaProp, nJobs)

    ### migration step (optional) ###
    if 'migration' in kwargs.keys():
      if step == migrationStart: status = 'Migration started at ' + str(migrationStart)
      elif step == migrationStop: status =  'Migration over at ' + str(migrationStop)

      #run hypermigration less frequently
      if (step > migrationStart) & (step < migrationStop) & (step % (kwargs['migration'] * 1) == 0):
        chains = range(nChains)
        shuffle(chains)
        newVals = Parallel(n_jobs = -1)(delayed(migrationHyper)(use_theta[:,chains,prm], use_phiMu[:,prm], use_phiSigma[:,prm], priors, dist, prm, p, rp * 10, kdePriors) for prm in xrange(len(p)))
        for param in xrange(len(p)): use_phiMu[:,param], use_phiSigma[:,param], use_hyperWeight[:,param] = newVals[param]      

      #subject migration
      if (step > migrationStart) & (step < migrationStop) & (step % kwargs['migration'] == 0):      
        chains = range(nChains)
        shuffle(chains)
        newVals = Parallel(n_jobs = -1)(delayed(migration)(use_theta[subj], use_phiMu[chains], use_phiSigma[chains], use_like[subj], data[data['subj'] == subject], log_dens_like, p, dist, rp * 10) for subj, subject in enumerate(np.unique(data['subj'])))
        #newVals = Parallel(n_jobs = -1)(delayed(migration)(use_theta[subj], use_phiMu, use_phiSigma, use_like[subj], data[data['subj'] == subject], log_dens_like, p, dist, rp) for subj, subject in enumerate(np.unique(data['subj'])))
        for subj in xrange(nSubj): use_theta[subj], use_like[subj], use_weight[subj] = newVals[subj]
    
    #update the data structures
    if (step >= burnin) & (step % thin == 0):
      phiMu[:,i], phiSigma[:,i], hyperWeight[:,i], theta[:,:,i], like[:,:,i], weight[:,:,i]  = use_phiMu, use_phiSigma, use_hyperWeight, use_theta, use_like, use_weight
      i += 1
      if i == nmc: MCMC = False

    #print progress
    '''
    if (step + 1) % 50 == 0:
      print '*',
    if (step + 1) % 1000 == 0:
      gc.collect()
      print step + 1'''
    if pb == True:
      if (step % 20 == 0) | (step == 1): update_progress(step, burnin + (nmc * thin), status)
    else:
      if (step + 1) % 1000 == 0: print '*',
      if (step + 1) % 10000 == 0: print step + 1

  if pb == True: update_progress(step, burnin + (nmc * thin), status = 'Finished')
  else: print 'Finished!'
  return theta, phiMu, phiSigma, like, weight, hyperWeight, priors

def samplingGroup(data, p, dist, log_dens_like, nChains = 4, nmc = 100, nJobs = 4, informedStart = -1, **kwargs):

  if 'rp' in kwargs.keys(): rp = kwargs['rp']
  else: rp = .001

  nParams, nSubj = len(p), len(np.unique(data['subj']))
  like, weight, theta = np.zeros((nChains, nmc), dtype = np.float32), np.zeros((nChains, nmc), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32)

  priors = generatePriors(p)
  priorMus = np.array([priors[param + '_mu'] for param in p])
  priorSigmas = np.array([priors[param + '_sigma'] for param in p])
  priors = [priorMus, priorSigmas]

  ### START VALUES
  if informedStart == -1:
    ### START VALUES
    for param in p:
      sigma = np.abs(starts[param] / 10.)
      if p.index(param) in dist['dtnorm']:
        theta[:,0,p.index(param)] = rtnorm(0, np.inf, starts[param], sigma, size = nChains)
      if 'dtnorm01' in dist.keys():
        if p.index(param) in dist['dtnorm01']:
          theta[:,0,p.index(param)] = rtnorm(0, 1.0, starts[param], sigma, size = nChains)
      if 'norm' in dist.keys():
        if p.index(param) in dist['norm']:
          theta[:,0,p.index(param)] = norm.rvs(starts[param], sigma, size = nChains)
    print 'Start values generated...'
  elif informedStart != -1:
    print 'Using informed start values...'
    preTheta, preWeight = getSubj(-1, informedStart), getWeight(informedStart)
    thetaStart = preTheta[np.unravel_index(preWeight.argmax(), preWeight.shape)]
    for param in p:
      start = thetaStart[p.index(param)]
      if p.index(param) in dist['dtnorm']:
        theta[:,0,p.index(param)] = rtnorm(0, np.inf, start, start / 10., size = nChains)
      if 'dtnorm01' in dist.keys():
        if p.index(param) in dist['dtnorm01']:
          theta[:,0,p.index(param)] = rtnorm(0, 1.0, start, start / 10., size = nChains)
      if 'norm' in dist.keys():
        if p.index(param) in dist['norm']:
          theta[:,0,p.index(param)] = norm.rvs(loc = start, scale = start / 10., size = nChains)

  print 'Start values generated...'

  like[:,0] = log_dens_like(theta[:,0], data, p)
  weight[:,0] = like[:,0] + log_dens_prior(theta[:,0], priors, dist)

  if np.mean(np.isfinite(weight[:,0])) == 1.0: print 'Check passed!'
  else: print 'Check failed - parameters illegal under priors...'

  print 'Beginning sampling...'
  for step in xrange(1, nmc):
    theta[:,step], like[:,step], weight[:,step] = crossoverGroup(like[:,step-1], weight[:,step-1], theta[:,step-1], priors, data, log_dens_like, p, dist, rp)

    if (step + 1) % 25 == 0: print '*',
    if (step + 1) % 500 == 0: print step + 1
  return theta, like, weight

def samplingHybrid(data, p, pHier, pGroup, dist, distHier, distGroup, log_dens_like, starts, priorsHier, priorsGroup, nChains = 4, nmc = 100, burnin = 0, thin = 1, nJobs = 4, informedStart = -1, cont = False, **kwargs):
 
  if burnin == 0: print '***NO BURN-IN***'

  #option of changing the perturbation parameter
  if 'rp' in kwargs.keys(): rp = kwargs['rp']
  else: rp = .001
  
  if 'minSigma' in kwargs.keys(): minSigma = kwargs['minSigma']
  else: minSigma = 0.0

  nParams, nSubj = len(p), len(np.unique(data['subj']))
  use_theta, use_phiMu, use_phiSigma = np.zeros((nSubj, nChains, len(p)), dtype = np.float32), np.zeros((nChains, len(p)), dtype = np.float32), np.zeros((nChains, len(p)), dtype = np.float32)

  #easy indexing for hier and group params
  hierParams = [p.index(param) for param in pHier]
  groupParams = [p.index(param) for param in pGroup]

  #have priors for both group and hierarchical parameters
  priorMuMus = np.array([priorsHier[param + '_mumu'] for param in pHier])
  priorMuSigmas = np.array([priorsHier[param + '_musigma'] for param in pHier])
  priorSigmaAlphas = np.array([priorsHier[param + '_sigmaalpha'] for param in pHier])
  priorSigmaBetas = np.array([priorsHier[param + '_sigmabeta'] for param in pHier])
  priorMus = np.array([priorsGroup[param + '_mu'] for param in pGroup])
  priorSigmas = np.array([priorsGroup[param + '_sigma'] for param in pGroup])
  priorsHier = [priorMuMus, priorMuSigmas, priorSigmaAlphas, priorSigmaBetas]
  priorsGroup = [priorMus, priorSigmas]

  #check if all parameters have priors in dist
  dParamsHier, dParamsGroup = [], []
  for key in distHier.keys(): dParamsHier += distHier[key]
  for key in distGroup.keys(): dParamsGroup += distGroup[key]
  dParamsHier.sort()
  dParamsGroup.sort()
  if (dParamsHier != range(len(pHier))) | (dParamsGroup != range(len(pGroup))):
    print 'xxx'
    print 'Parameters NOT all represented in dist'
    print 'xxx'
    print 'distHier: ' + str(dParamsHier) + 'pHier: ' + str(range(len(pHier)))
    print 'distGroup: ' + str(dParamsGroup) + ' pGroup: ' + str(range(len(pGroup)))

  ##### START VALUES #####

  ###randomly generated start values
  #...these scale factors denote how sigma values are scaled relative to the means
  sigmaScale = 10.
  phiSigmaScale = 10.
  sigmaSigmaScale = 20.
  
  #generate start values from scratch
  if informedStart == -1:
    print 'Creating values...'
    for param in pHier:
      if starts[param] != 0.0: sigma, phiSigmaStart, sigmaSigma = np.abs(float(starts[param]) / sigmaScale), np.min(np.abs(float(starts[param])), .05), np.abs(float(starts[param]) / sigmaSigmaScale)
      #if starts[param] != 0.0: sigma, phiSigmaStart, sigmaSigma = np.abs(float(starts[param]) / sigmaScale), .3, np.abs(float(starts[param]) / sigmaSigmaScale)
         
      elif starts[param] == 0.0: sigma, sigmaSigma = .001, .001
      use_phiSigma[:,p.index(param)] = rtnorm(minSigma, np.inf, phiSigmaStart, sigmaSigma, size = nChains)
      
      if p.index(param) in dist['dtnorm']:
        use_phiMu[:,p.index(param)] = rtnorm(0, np.inf, starts[param], sigma, size = nChains)
        use_theta[:,:,p.index(param)] = rtnorm(0, np.inf, use_phiMu[:,p.index(param)].reshape(1,nChains), use_phiSigma[:,p.index(param)].reshape(1,nChains) / phiSigmaScale, size = (nSubj, nChains))
      if 'dtnorm01' in dist.keys():
        if p.index(param) in dist['dtnorm01']:
          use_phiMu[:,p.index(param)] = rtnorm(0.0, 1.0, starts[param], sigma, size = nChains)
          use_theta[:,:,p.index(param)] = rtnorm(0.0, 1.0, use_phiMu[:,p.index(param)], use_phiSigma[:,p.index(param)] / phiSigmaScale, size = (nSubj,nChains))
      if 'norm' in dist.keys():
        if p.index(param) in dist['norm']: 
          use_phiMu[:,p.index(param)] = norm.rvs(starts[param], sigma, size = nChains)
          use_theta[:,:,p.index(param)] = norm.rvs(use_phiMu[:,p.index(param)].reshape(1,nChains), use_phiSigma[:,p.index(param)].reshape(1,nChains) / phiSigmaScale, size = (nSubj, nChains))
    for param in pGroup:
      if p.index(param) in dist['dtnorm']:
        use_phiMu[:,p.index(param)] = rtnorm(0, np.inf, starts[param], starts[param] / sigmaScale, size = nChains)
      if 'dtnorm01' in dist.keys():
        if p.index(param) in dist['dtnorm01']:
          use_phiMu[:,p.index(param)] = rtnorm(0.0, 1.0, starts[param], starts[param] / sigmaScale, size = nChains)   
      if 'norm' in dist.keys():
        if p.index(param) in dist['norm']: 
          use_phiMu[:,p.index(param)] = norm.rvs(starts[param], sigma, size = nChains)
      use_theta[:,:,p.index(param)] = use_phiMu[:,p.index(param)].reshape(1,nChains) #all subjects take on the group value
          
  #use previous fits
  elif informedStart != -1:
    if cont == False:
      print 'Loading start values from a previous fit...'
      use_phiMu = getPhiMu(informedStart)[0:nChains,-1]
      use_phiSigma = getPhiSigma(informedStart)[0:nChains,-1]
      for subj in xrange(nSubj): use_theta[subj,:] = getSubj(subj,informedStart)[0:nChains,-1]
      use_theta[:,:,groupParams] = use_phiMu[:,groupParams].reshape(1,nChains,len(groupParams)) #this makes it such that a fully hierarchical fit can be continued
    elif cont == True:
      print 'Continuing a previous fit...'
      theta0 = getSubj(0, informedStart)
      nChains, nPrevious, nParams = theta0.shape
      theta = np.zeros((nSubj,nChains,nPrevious + nmc,nParams))
      theta[0] = np.concatenate((theta0, np.zeros((nChains,nmc,nParams))), axis = 1)
      for subj in xrange(1,nSubj):
        theta[subj] = np.concatenate((getSubj(subj,informedStart), np.zeros((nChains,nmc,nParams))), axis = 1)

      phiMu = np.concatenate((getPhiMu(informedStart), np.zeros((nChains,nmc,nParams))), axis = 1)
      phiSigma = np.concatenate((getPhiMu(informedStart), np.zeros((nChains,nmc,nParams))), axis = 1)
      weight = np.concatenate((getWeight(informedStart), np.zeros((nSubj,nChains,nmc))), axis = 2)
      hyperWeight = np.concatenate((getHyperWeight(informedStart), np.zeros((nChains,nmc,len(hierParams)))), axis = 1)
      groupWeight = np.concatenate((getGroupWeight(informedStart), np.zeros((nChains,nmc))), axis = 1)

      del theta0
      gc.collect()

      use_theta, use_phiMu, use_phiSigma = theta[:,:,nPrevious-1], phiMu[:,nPrevious-1], phiSigma[:,nPrevious-1]
    
  print 'Start values generated...'

  ### generate the initial set of likelihoods
  use_like = np.array(Parallel(n_jobs = nJobs)(delayed(log_dens_like)(use_theta[subj], data[data['subj'] == subject], p) for subj, subject in enumerate(np.unique(data['subj']))))
  prior = log_dens_hyper(use_theta[:,:,hierParams], use_phiMu[:,hierParams], use_phiSigma[:,hierParams], distHier)
  use_weight = prior + use_like
  use_hyperWeight = log_dens_hyper_and_prior(use_theta[:,:,hierParams], use_phiMu[:,hierParams], use_phiSigma[:,hierParams], priorsHier, distHier)
  use_groupWeight = log_dens_prior(use_phiMu[:,groupParams], priorsGroup, distGroup) + np.sum(use_like, axis = 0)

  ###check if parameters are ok
  if (np.mean(np.isfinite(np.mean(use_weight))) == 1.0) & (np.mean(np.isfinite(use_hyperWeight)) == 1.0) & (np.mean(np.isfinite(use_groupWeight)) == 1.0): print 'Check passed!'
  else:
    print 'Check failed...'
    if np.mean(np.isfinite(np.mean(use_like))) != 1.0:
      print 'Bad likelihood values...'
      print np.mean(np.isfinite(use_like))

    if np.mean(np.isfinite(prior)) != 1.0:
      print 'Subject parameters illegal under hyperparameters...'      
    if np.mean(np.isfinite(use_hyperWeight)) != 1.0:
      print 'Hyperparameters illegal under priors...'
      print np.mean(np.isfinite(use_hyperWeight))
    if np.mean(np.isfinite(use_groupWeight)) != 1.0:
      print 'Group parameters illegal under priors...'
      print np.mean(np.isfinite(use_groupWeight)) 

  #replace nans with negative inf
  use_weight[np.isnan(use_weight)] = -1 * np.inf
  use_like[np.isnan(use_like)] = -1 * np.inf
 
  #gamma
  gammaProp = 2.38 / np.sqrt(2 * len(pHier))
  #gammaGroup = 2.38 / np.sqrt(2 * len(pGroup))
  #gammaHyper = 2.38 / np.sqrt(2 * 2)

  print 'Beginning sampling...'
  if 'migrationStart' in kwargs.keys(): migrationStart = kwargs['migrationStart']
  else: migrationStart = 15000
  if 'migrationStop' in kwargs.keys(): migrationStop = kwargs['migrationStop']
  else: migrationStop = 10**1000
  gc.collect()

  #create structures right away if burnin is zero
  if (burnin == 0) & (cont == False):
    theta, phiMu, phiSigma = np.zeros((nSubj, nChains, nmc, nParams), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32)
    hyperWeight, weight, groupWeight = np.zeros((nChains, nmc, len(pHier)), dtype = np.float32), np.zeros((nSubj, nChains, nmc), dtype = np.float32), np.zeros((nChains, nmc), dtype = np.float32)
    phiMu[:,0], phiSigma[:,0], hyperWeight[:,0], theta[:,:,0], weight[:,:,0] = use_phiMu, use_phiSigma, use_hyperWeight, use_theta, use_weight
    i = 0

  ### MCMC loop ###
  MCMC = True
  step = 0
  
  if cont == True:
    step = nPrevious
    i = nPrevious #ensure that MCMC starts where it left off
    nmc += nPrevious
  print str(step)
  
  divisor = len(groupParams)
  if 'divisor' in kwargs.keys():
    if kwargs['divisor'] == 'sqrt': divisor = np.sqrt(2 * len(groupParams))
    elif kwargs['divisor'] == None: divisor = 1

  while MCMC == True:

    gammaGroup = np.random.uniform(.5, 1) / divisor
    gammaHyper = np.random.uniform(.5, 1)

    step += 1

    ### create the data structures if burnin period is over
    if (step == burnin) & (cont == False):
      print 'Burnin period over... recording samples...'
      theta, phiMu, phiSigma = np.zeros((nSubj, nChains, nmc, nParams), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32)
      hyperWeight, weight, groupWeight = np.zeros((nChains, nmc, len(pHier)), dtype = np.float32), np.zeros((nSubj, nChains, nmc), dtype = np.float32), np.zeros((nChains, nmc))    
      i = 0

    ### crossover hyper step: sample new hyperparameters ###
    use_phiMu[:,hierParams], use_phiSigma[:,hierParams], use_hyperWeight = crossoverHyper(gammaHyper, use_theta[:,:,hierParams], use_phiMu[:,hierParams], use_phiSigma[:,hierParams], priorsHier, distHier, p, rp, kdePriors)

    ### sample new group parameters ###
    use_phiMu[:,groupParams], use_theta, use_groupWeight, use_like = crossoverHybrid_group(gammaGroup, use_like, use_theta, use_phiMu[:,groupParams], data, log_dens_like, p, groupParams, hierParams, distGroup, priorsGroup, rp, nJobs)

    ### crossover step: sample new subject parameters ###
    #consider just slicing out the use_phiMu[:,hierParams]
    use_theta, use_like, use_weight = crossoverHybrid_subj(gammaProp, use_like, use_theta, use_phiMu, use_phiSigma, data, log_dens_like, p, hierParams, groupParams, distHier, rp, nJobs)

    ### migration step (optional) ###
    if 'migration' in kwargs.keys():
      if step == migrationStart: print 'Migration beginning!'
      elif step == migrationStop: print 'Migration ceasing!'
      if (step >= migrationStart) & (step < migrationStop) & (step % kwargs['migration'] == 0):
        #indexing below is tricky... note that theta/phiMu are indexed by param (absolute indexing) and prm is the relative index (which indexes priors/dist later)
        newVals = Parallel(n_jobs = -1)(delayed(migrationHyper)(use_theta[:,:,param], use_phiMu[:,param], use_phiSigma[:,param], priorsHier, distHier, prm, rp) for prm, param in enumerate(hierParams))
        for prm, param in enumerate(hierParams): use_phiMu[:,param], use_phiSigma[:,param], use_hyperWeight[:,prm] = newVals[prm]   

        use_phiMu[:,groupParams], use_theta, use_groupWeight, use_like = migrationHybrid_group(use_theta, use_phiMu[:,groupParams], use_like, data, log_dens_like, priorsGroup, p, groupParams, hierParams, distGroup, rp, nJobs)

        before = np.mean(use_weight)
        newVals = Parallel(n_jobs = -1)(delayed(migrationHybrid_subj)(use_theta[subj], use_phiMu[:,hierParams], use_phiSigma[:,hierParams], use_like[subj], data[data['subj'] == subject], log_dens_like, p, hierParams, groupParams, distHier, rp) for subj, subject in enumerate(np.unique(data['subj'])))
        for subj in xrange(nSubj): use_theta[subj], use_like[subj], use_weight[subj] = newVals[subj]

    #update the data structures
    if (step >= burnin) & (step % thin == 0):
      phiMu[:,i], phiSigma[:,i], hyperWeight[:,i], theta[:,:,i], weight[:,:,i], groupWeight[:,i] = use_phiMu, use_phiSigma, use_hyperWeight, use_theta, use_weight, use_groupWeight
      
      i += 1
      if i == nmc: MCMC = False

    #print progress
    if (step + 1) % 25 == 0:
      print '*',
    if (step + 1) % 500 == 0:
      gc.collect()
      print step + 1

  print 'Finished...'
  #/end of MCMC loop
  return theta, phiMu, phiSigma, weight, hyperWeight, groupWeight

def samplingGroup(data, p, dist, log_dens_like, nChains = 4, nmc = 100, nJobs = 4, informedStart = -1, **kwargs):

  if 'rp' in kwargs.keys(): rp = kwargs['rp']
  else: rp = .001

  nParams, nSubj = len(p), len(np.unique(data['subj']))
  like, weight, theta = np.zeros((nChains, nmc), dtype = np.float32), np.zeros((nChains, nmc), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32)

  priors = generatePriors(p)
  priorMus = np.array([priors[param + '_mu'] for param in p])
  priorSigmas = np.array([priors[param + '_sigma'] for param in p])
  priors = [priorMus, priorSigmas]

  ### START VALUES
  if informedStart == -1:
    ### START VALUES
    for param in p:
      sigma = np.abs(starts[param] / 10.)
      if p.index(param) in dist['dtnorm']:
        theta[:,0,p.index(param)] = rtnorm(0, np.inf, starts[param], sigma, size = nChains)
      if 'dtnorm01' in dist.keys():
        if p.index(param) in dist['dtnorm01']:
          theta[:,0,p.index(param)] = rtnorm(0, 1.0, starts[param], sigma, size = nChains)
      if 'norm' in dist.keys():
        if p.index(param) in dist['norm']:
          theta[:,0,p.index(param)] = norm.rvs(starts[param], sigma, size = nChains)
    print 'Start values generated...'
  elif informedStart != -1:
    print 'Using informed start values...'
    preTheta, preWeight = getSubj(-1, informedStart), getWeight(informedStart)
    thetaStart = preTheta[np.unravel_index(preWeight.argmax(), preWeight.shape)]
    for param in p:
      start = thetaStart[p.index(param)]
      if p.index(param) in dist['dtnorm']:
        theta[:,0,p.index(param)] = rtnorm(0, np.inf, start, start / 10., size = nChains)
      if 'dtnorm01' in dist.keys():
        if p.index(param) in dist['dtnorm01']:
          theta[:,0,p.index(param)] = rtnorm(0, 1.0, start, start / 10., size = nChains)
      if 'norm' in dist.keys():
        if p.index(param) in dist['norm']:
          theta[:,0,p.index(param)] = norm.rvs(loc = start, scale = start / 10., size = nChains)

  print 'Start values generated...'

  like[:,0] = log_dens_like(theta[:,0], data, p)
  weight[:,0] = like[:,0] + log_dens_prior(theta[:,0], priors, dist)

  if np.mean(np.isfinite(weight[:,0])) == 1.0: print 'Check passed!'
  else: print 'Check failed - parameters illegal under priors...'

  print 'Beginning sampling...'
  for step in xrange(1, nmc):
    theta[:,step], like[:,step], weight[:,step] = crossoverGroup(like[:,step-1], weight[:,step-1], theta[:,step-1], priors, data, log_dens_like, p, dist, rp)

    if (step + 1) % 25 == 0: print '*',
    if (step + 1) % 500 == 0: print step + 1
  return theta, like, weight

#this is used for a single subject
def samplingSubject(data, p, dist, log_dens_like, starts, priors, subj, nChains = 4, nmc = 100, burnin = 0, thin = 1, nChainBlocks = 4, nJobs = 4, informedStart = -1, cont = False, **kwargs):

  print '-----------'
  print 'SUBJECT FIT'
  print '-----------'
  print 'nChains: ' + str(nChains) + ' Burnin: ' + str(burnin) + ' nmc: ' + str(nmc) + ' thin: ' + str(thin)
  print '# of parameters: ' + str(len(p))
  print p
  print 'dist: ' + str(dist)

  if 'gamma1' in kwargs.keys(): gamma1 = kwargs['gamma1']
  else: gamma1 = False
  if 'gammat0' in kwargs.keys(): gammat0 = kwargs['gammat0']
  else: gammat0 = False  
  if 'gammav' in kwargs.keys(): gammav = kwargs['gammav']
  else: gammav = False
  
  if gamma1 == True: print 'Using gamma = 1 every 10th iteration'
  if gammat0 != False: print 'Using gamma = 1.1 for t0/st0 for every 10th iteration'
  if gammav != False: print 'Using gamma = 1.1 for Vti/Vtt/Vaa/Vab for every 10th iteration'

  if burnin == 0: print '***NO BURN-IN***'

  #option of changing the perturbation parameter
  if 'rp' in kwargs.keys(): rp = kwargs['rp']
  else: rp = .001

  data = data[data['subj'] == np.unique(data['subj'])[subj]]
  print 'Fitting subject # ' + str(subj)

  #option of changing the perturbation parameter
  if 'rp' in kwargs.keys(): rp = kwargs['rp']
  else: rp = .001
  
  if 'minSigma' in kwargs.keys(): minSigma = kwargs['minSigma']
  else: minSigma = 0.0
  
  #gammaUnif: gamma = uniform(.5, 1)
  if 'gammaUnif' in kwargs.keys():
    gammaUnif = kwargs['gammaUnif']
    if gammaUnif == True: print 'Gamma sampled from Uniform(.5, 1.0)'
  else: gammaUnif = False

  nParams, nSubj = len(p), len(np.unique(data['subj']))
  use_theta = np.zeros((nChains, len(p)), dtype = np.float32)

  priorMus = np.array([priors[param + '_mu'] for param in p])
  priorSigmas = np.array([priors[param + '_sigma'] for param in p])
  priors = [priorMus, priorSigmas]
  ##### START VALUES #####

  ###randomly generated start values
  #...these scale factors denote how sigma values are scaled relative to the means
  sigmaScale = 7.0

  if informedStart == -1:
    print 'Creating values...'
    for param in p:
      if starts[param] != 0.0: sigma = np.abs(float(starts[param]) / sigmaScale)
      elif starts[param] == 0.0: sigma, sigmaSigma = .01, .01

      if p.index(param) in dist['dtnorm']:
        use_theta[:,p.index(param)] = rtnorm(0, np.inf, starts[param], sigma, size = nChains)
      if 'dtnorm01' in dist.keys():
        if p.index(param) in dist['dtnorm01']:
          use_theta[:,p.index(param)] = rtnorm(0.0, 1.0, starts[param], sigma, size = nChains)
      if 'norm' in dist.keys():
        if p.index(param) in dist['norm']: 
          use_theta[:,p.index(param)] = norm.rvs(starts[param], sigma, size = nChains)
          
  elif (informedStart != -1):
    print 'Continuing a previous fit...'
    use_theta = getSubj(subj,informedStart)[0:nChains,-1]

  print 'Start values generated...'

  ### generate the initial set of likelihoods
  cB = nChains / nChainBlocks
  use_like = np.array(Parallel(n_jobs = -1)(delayed(log_dens_like)(use_theta[cB*c:cB*(c+1)], data, p) for c in xrange(nChainBlocks))).flatten()
  prior = log_dens_prior(use_theta, priors, dist)

  use_weight = prior + use_like

  ###check if parameters are ok
  if (np.mean(np.isfinite(np.mean(use_weight))) == 1.0): print 'Check passed!'
  else:
    print 'Check failed...'
    if np.mean(np.isfinite(np.mean(use_like))) != 1.0:
      print 'Bad likelihood values...'
      print np.mean(np.isfinite(use_like))

    if np.mean(np.isfinite(prior)) != 1.0:
      print 'Subject parameters illegal under hyperparameters...'      

  #replace nans with negative inf
  use_weight[np.isnan(use_weight)] = -1 * np.inf
  use_like[np.isnan(use_like)] = -1 * np.inf
 
  #gamma
  gammaProp = 2.38 / np.sqrt(2 * len(p))

  print 'Beginning sampling...'
  if 'migrationStart' in kwargs.keys(): migrationStart = kwargs['migrationStart']
  else: migrationStart = 15000
  if 'migrationStop' in kwargs.keys(): migrationStop = kwargs['migrationStop']
  else: migrationStop = 10**1000
  gc.collect()

  #create structures right away if burnin is zero: only do this if not continuing a fit
  if (burnin == 0) & (cont == False):
    theta = np.zeros((nChains, nmc, nParams), dtype = np.float32)
    like = np.zeros((nChains, nmc), dtype = np.float32)
    weight = np.zeros((nChains, nmc), dtype = np.float32)
    theta[:,0] = use_theta
    like[:,0] = use_like
    weight[:,0] = use_weight
    i = 0

  ### MCMC loop ###
  MCMC = True
  step = 0
  rp1 = rp

  while MCMC == True:

    step += 1

    if ((gamma1 == True) | (step < burnin)) & ((step + 5) % 10 == 0):
      gammaProp = .98
    #elif ((gamma1 == True) | (step < burnin)) & ((step + 7) % 10 == 0): gammaProp = np.random.uniform(.5, 1.0, size = (nChains,1))
    elif (gammat0 != False) & (step % 10 == 0):
      t = [p.index(param) for param in p if ('t0' in param) | ('st0' in param)] #just upscale the problem parameters
      if gammaUnif == True: gammaProp = np.ones((nChains,nParams)) * np.random.uniform(.5, 1.0, size = (nChains,1))
      else: gammaProp = np.ones((1,nParams)) * (2.38 / np.sqrt(2 * nParams))
      if gammat0 == True: gammaVal = 1.1
      else: gammaVal = kwargs['gammat0']
      gammaProp[:,t] = gammaVal
    elif (gammav != False) & (step % 10 == 0):
      #source memory specific one
      t = [p.index(param) for param in p if ('logVti' in param) | ('logVtt' in param) | ('logVaa' in param) | ('logVab' in param)] #just upscale the problem parameters
      if gammaUnif == True: gammaProp = np.ones((nChains,nParams)) * np.random.uniform(.5, 1.0, size = (nChains,1))
      else: gammaProp = np.ones((1,nParams)) * (2.38 / np.sqrt(2 * nParams))
      if gammat0 == True: gammaVal = 1.1
      else: gammaVal = kwargs['gammav']
      gammaProp[:,t] = gammaVal
    elif gammaUnif == True: gammaProp = np.random.uniform(.5, 1.0, size = (nChains,1))
    else: gammaProp = 2.38 / np.sqrt(2 * nParams)
    #gammaProp = np.random.uniform(.5, 1)
    if (step < burnin) | (step % 15 == 0): rp = rp1 * 5
    else: rp = rp1

    ### create the data structures if burnin period is over
    if step == burnin:
      print 'Burnin period over... recording samples...'
      theta = np.zeros((nChains, nmc, nParams), dtype = np.float32)
      like = np.zeros((nChains, nmc), dtype = np.float32)
      weight = np.zeros((nChains, nmc), dtype = np.float32)
      i = 0

    ### crossover step: sample new subject parameters ###
    #consider just slicing out the use_phiMu[:,hierParams]
    use_theta, use_like, use_weight = crossoverSubject(gammaProp, use_like, use_theta, priors, data, log_dens_like, p, dist, rp, nChainBlocks, nJobs)

    ### migration step (optional) ###
    if 'migration' in kwargs.keys():
      if step == migrationStart: print 'Migration beginning!'
      elif step == migrationStop: print 'Migration ceasing!'
      if (step >= migrationStart) & (step < migrationStop) & (step % kwargs['migration'] == 0):
        use_theta, use_like, use_weight = migrationSubject(use_theta, use_like, data, log_dens_like, p, dist, priors, rp, nJobs = nJobs)
    
    #update the data structures
    if (step >= burnin) & (step % thin == 0):
      theta[:,i], like[:,i], weight[:,i] = use_theta, use_like, use_weight
      
      i += 1
      if i == nmc: MCMC = False

    #print progress
    if (step + 1) % 25 == 0:
      print '*',
    if (step + 1) % 500 == 0:
      gc.collect()
      print step + 1

  print 'Finished...'
  #/end of MCMC loop
  return theta, like, weight, priors

#this is used for MULTIPLE subjects
def samplingSubjects(data, p, dist, log_dens_like, starts, priors, nChains = 4, nmc = 100, burnin = 0, thin = 1, nChainBlocks = 4, nJobs = 4, informedStart = -1, cont = False, **kwargs):

  print '---------------------'
  print 'SUBJECTS (plural) FIT'
  print '---------------------'
  print 'nChains: ' + str(nChains) + ' Burnin: ' + str(burnin) + ' nmc: ' + str(nmc) + ' thin: ' + str(thin)
  print '# of parameters: ' + str(len(p))
  print p
  print 'dist: ' + str(dist)

  if 'gamma1' in kwargs.keys(): gamma1 = kwargs['gamma1']
  else: gamma1 = False
  if 'gammat0' in kwargs.keys(): gammat0 = kwargs['gammat0']
  else: gammat0 = False  
  
  if gamma1 == True: print 'Using gamma = 1 every 10th iteration'
  if gammat0 != False: print 'Using gamma = 1 for t0/st0 for every 10th iteration'

  if burnin == 0: print '***NO BURN-IN***'

  #option of changing the perturbation parameter
  if 'rp' in kwargs.keys(): rp = kwargs['rp']
  else: rp = .001

  #option of changing the perturbation parameter
  if 'rp' in kwargs.keys(): rp = kwargs['rp']
  else: rp = .001
  
  if 'minSigma' in kwargs.keys(): minSigma = kwargs['minSigma']
  else: minSigma = 0.0
  
  #gammaUnif: gamma = uniform(.5, 1)
  if 'gammaUnif' in kwargs.keys():
    gammaUnif = kwargs['gammaUnif']
    if gammaUnif == True: print 'Gamma sampled from Uniform(.5, 1.0)'
  else: gammaUnif = False

  nParams, nSubj = len(p), len(np.unique(data['subj']))
  use_theta = np.zeros((nSubj, nChains, len(p)), dtype = np.float32)

  priorMus = np.array([priors[param + '_mu'] for param in p])
  priorSigmas = np.array([priors[param + '_sigma'] for param in p])
  priors = [priorMus, priorSigmas]
  ##### START VALUES #####

  ###randomly generated start values
  #...these scale factors denote how sigma values are scaled relative to the means
  sigmaScale = 8.5

  if informedStart == -1:
    print 'Creating values...'
    for param in p:
      if type(starts[param]) == tuple:
        start = starts[param][0]
        sigma = starts[param][1]
      else:
        start = starts[param]
        if start != 0.0: sigma = np.abs(float(starts[param]) / sigmaScale)
        elif start == 0.0: sigma = .01

      if p.index(param) in dist['dtnorm']:
        use_theta[:,:,p.index(param)] = rtnorm(0, np.inf, start, sigma, size = nChains)
      if ('dtnorm01' in dist.keys()) | ('beta' in dist.keys()):
        if (p.index(param) in dist['dtnorm01']) | (p.index(param) in dist['beta']):
          use_theta[:,:,p.index(param)] = rtnorm(0.0, 1.0, start, sigma, size = nChains)
      if 'norm' in dist.keys():
        if p.index(param) in dist['norm']: 
          use_theta[:,:,p.index(param)] = norm.rvs(start, sigma, size = nChains)
          
  elif (informedStart != -1) & (cont == True):
    print 'Continuing a previous fit...'
    theta0 = getSubj(0, informedStart)[0:nChains]
    nChains, nPrevious, nParams = theta0.shape
    theta = np.zeros((nSubj,nChains,nPrevious,nParams))
    theta[0] = theta0
    for subj in xrange(1,nSubj): theta[subj] = getSubj(subj,informedStart)[0:nChains]

    theta = np.concatenate((theta, np.zeros((nSubj,nChains,nmc,nParams))), axis = 2)
    like = np.concatenate((getLike(informedStart)[:,0:nChains], np.zeros((nSubj,nChains,nmc))), axis = 2)
    weight = np.concatenate((getWeight(informedStart)[:,0:nChains], np.zeros((nSubj,nChains,nmc))), axis = 2)

    use_theta = theta[:,:,nPrevious-1]
    
    del theta0
    gc.collect()

  print 'Start values generated...'

  ### generate the initial set of likelihoods
  cB = nChains / nChainBlocks
  use_like = np.array(Parallel(n_jobs = nJobs)(delayed(log_dens_like)(use_theta[subj], data[data['subj'] == subject], p) for subj, subject in enumerate(np.unique(data['subj']))))
  prior = log_dens_priorSubjects(use_theta, priors, dist)

  use_weight = prior + use_like

  ###check if parameters are ok
  if (np.mean(np.isfinite(np.mean(use_weight))) == 1.0): print 'Check passed!'
  else:
    print 'Check failed...'
    if np.mean(np.isfinite(np.mean(use_like))) != 1.0:
      print 'Bad likelihood values...'
      print np.mean(np.isfinite(use_like))

    if np.mean(np.isfinite(prior)) != 1.0:
      print 'Subject parameters illegal under hyperparameters...'      

  #replace nans with negative inf
  use_weight[np.isnan(use_weight)] = -1 * np.inf
  use_like[np.isnan(use_like)] = -1 * np.inf
 
  #gamma
  gammaProp = 2.38 / np.sqrt(2 * len(p))

  print 'Beginning sampling...'
  if 'migrationStart' in kwargs.keys(): migrationStart = kwargs['migrationStart']
  else: migrationStart = 15000
  if 'migrationStop' in kwargs.keys(): migrationStop = kwargs['migrationStop']
  else: migrationStop = 10**1000
  gc.collect()

  #create structures right away if burnin is zero: only do this if not continuing a fit
  if (burnin == 0) & (cont == False):
    theta = np.zeros((nSubj, nChains, nmc, nParams), dtype = np.float32)
    like = np.zeros((nSubj, nChains, nmc), dtype = np.float32)
    weight = np.zeros((nSubj, nChains, nmc), dtype = np.float32)
    theta[:,:,0] = use_theta
    like[:,:,0] = use_like
    weight[:,:,0] = use_weight
    i = 0

  ### MCMC loop ###
  MCMC = True
  step = 0
  rp1 = rp

  if cont == True:
    step = nPrevious
    i = nPrevious #ensure that MCMC starts where it left off
    nmc += nPrevious
  print str(step)

  while MCMC == True:

    step += 1

    if ((gamma1 == True) | (step < burnin)) & ((step + 5) % 10 == 0):
      gammaProp = .98
    elif ((gamma1 == True) | (step < burnin)) & ((step + 7) % 10 == 0):
      gammaProp = np.random.uniform(.5, 1.0, size = (nChains,1))
    elif (gammat0 != False) & (step % 10 == 0):
      t = [p.index(param) for param in p if ('t0' in param) | ('st0' in param)] #just upscale the problem parameters
      if gammaUnif == True: gammaProp = np.ones((nChains,nParams)) * np.random.uniform(.5, 1.0, size = (nChains,1))
      else: gammaProp = np.ones((1,nParams)) * (2.38 / np.sqrt(2 * nParams))
      if gammat0 == True: gammaVal = 1.1
      else: gammaVal = kwargs['gammat0']
      gammaProp[:,t] = gammaVal
    elif gammaUnif == True: gammaProp = np.random.uniform(.5, 1.0, size = (nChains,1))
    else: gammaProp = 2.38 / np.sqrt(2 * nParams)
    #gammaProp = np.random.uniform(.5, 1)
    if (step < burnin) | (step % 15 == 0): rp = rp1 * 5
    else: rp = rp1

    ### create the data structures if burnin period is over
    if step == burnin:
      print 'Burnin period over... recording samples...'
      theta = np.zeros((nSubj, nChains, nmc, nParams), dtype = np.float32)
      like = np.zeros((nSubj, nChains, nmc), dtype = np.float32)
      weight = np.zeros((nSubj, nChains, nmc), dtype = np.float32)
      i = 0

    ### crossover step: sample new subject parameters ###
    #consider just slicing out the use_phiMu[:,hierParams]
    use_theta, use_like, use_weight = crossoverSubjects(gammaProp, use_like, use_theta, priors, data, log_dens_like, p, dist, rp, nJobs)

    ### migration step (optional) ###
    if 'migration' in kwargs.keys():
      if step == migrationStart: print 'Migration beginning!'
      elif step == migrationStop: print 'Migration ceasing!'
      if (step > migrationStart) & (step < migrationStop) & (step % kwargs['migration'] == 0):      
        chains = range(nChains)
        shuffle(chains)
        newVals = Parallel(n_jobs = -1)(delayed(migrationSubject)(use_theta[subj], use_like[subj], data[data['subj'] == subject], log_dens_like, p, dist, priors, rp * 10, parallel = False) for subj, subject in enumerate(np.unique(data['subj'])))
        #newVals = Parallel(n_jobs = -1)(delayed(migration)(use_theta[subj], use_phiMu, use_phiSigma, use_like[subj], data[data['subj'] == subject], log_dens_like, p, dist, rp) for subj, subject in enumerate(np.unique(data['subj'])))
        for subj in xrange(nSubj): use_theta[subj], use_like[subj], use_weight[subj] = newVals[subj]
    
    #update the data structures
    if (step >= burnin) & (step % thin == 0):
      theta[:,:,i], like[:,:,i], weight[:,:,i] = use_theta, use_like, use_weight
      
      i += 1
      if i == nmc: MCMC = False

    #print progress
    if (step + 1) % 25 == 0:
      print '*',
    if (step + 1) % 500 == 0:
      gc.collect()
      print step + 1

  print 'Finished...'
  #/end of MCMC loop
  return theta, like, weight, priors

def calculateDIC(data, params, log_dens_like, theta, like, BPIC = False):
  nSubj = len(np.unique(data['subj']))

  meanLike = np.mean(np.sum(like, axis = 0))
  meanTheta = np.array([np.mean(theta[subj], axis = (0,1)) for subj in xrange(nSubj)])

  likeAtMean = np.sum(np.array(Parallel(n_jobs = -1)(delayed(log_dens_like)(meanTheta[subj], data[data['subj'] == subject], params) for subj, subject in enumerate(np.unique(data['subj'])))))
  
  D = -2 * meanLike
  pD = D + (2 * likeAtMean)
  DIC = D + pD
  return D, pD, DIC

def samplingHierBetween(data, ps, hp, dists, distHiers, distHierAll, nSubjs, log_dens_like, starts, priors, nChains = 4, nmc = 100, burnin = 0, thin = 1, nJobs = 4, informedStart = -1, cont = False, recalculate = False, **kwargs):

  print '--------------------'
  print 'BETWEEN SUBJECTS FIT'
  print '--------------------'
  print 'nChains: ' + str(nChains) + ' Burnin: ' + str(burnin) + ' nmc: ' + str(nmc) + ' thin: ' + str(thin)
  print '# of hyper parameters: ' + str(len(hp))
  print '# of subject parameters in each group: ' + str([len(p) for p in ps])
  print hp
  for g in xrange(len(ps)):
    print 'params group ' + str(g), ps[g]
    print 'dist group ' + str(g), dists[g], 'dist hier group ' + str(g), distHiers[g]
  print 'dist all: ' + str(distHierAll)
  nParams = len(hp)
  
  if 'gamma1' in kwargs.keys(): gamma1 = kwargs['gamma1']
  else: gamma1 = False
  if 'gammat0' in kwargs.keys(): gammat0 = kwargs['gammat0']
  else: gammat0 = False
  if 'gammaVaa' in kwargs.keys(): gammaVaa = kwargs['gammaVaa']
  else: gammaVaa = False
  if 'gammar' in kwargs.keys(): gammar = kwargs['gammar']
  else: gammar = False
  if 'gammabgNoise' in kwargs.keys(): gammabgNoise = kwargs['gammabgNoise']
  else: gammabgNoise = False
  
  if gamma1 == True: print 'Using gamma = 1 every 10th iteration'
  elif gammat0 == True: print 'Using gamma = 1 for t0/st0 for every 10th iteration'
  if gammaVaa != False: print 'Using gamma = 1.1 for Vaa/Vab for every 10th iteration'
  if gammar != False: print 'Using gamma = 1.1 for r for every 10th iteration'
  if gammabgNoise != False: print 'Using gamma = 1.1 for bgNoise for every 20th iteration'

  #gammaUnif: gamma = uniform(.5, 1)
  if 'gammaUnif' in kwargs.keys():
    gammaUnif = kwargs['gammaUnif']
    if gammaUnif == True: print 'Gamma sampled from Uniform(.5, 1.0)'
  else: gammaUnif = False

  #message about recalculating likelihoods
  if recalculate != False: print 'Recalculating likelihoods every ', recalculate, ' samples. Welcome to PDA Land!!!'

  if burnin == 0: print '***NO BURN-IN***'

  #option of changing the perturbation parameter
  if 'rp' in kwargs.keys(): rp = kwargs['rp']
  else: rp = .001
  
  if 'minSigma' in kwargs.keys(): minSigma = kwargs['minSigma']
  else: minSigma = 0.0

  nSubj = len(np.unique(data['subj']))
  allPriors = []

  use_phiMu, use_phiSigma = np.zeros((nChains, len(hp)), dtype = np.float32), np.zeros((nChains, len(hp)), dtype = np.float32)

  #loop through each parameter list and create the relevant data structures
  priorMuMus = np.array([priors[param + '_mumu'] for param in hp])
  priorMuSigmas = np.array([priors[param + '_musigma'] for param in hp])
  priorSigmaAlphas = np.array([priors[param + '_sigmaalpha'] for param in hp])
  priorSigmaBetas = np.array([priors[param + '_sigmabeta'] for param in hp])
  priors = [priorMuMus, priorMuSigmas, priorSigmaAlphas, priorSigmaBetas]

  #if there are KDE priors, designate them now
  if 'kdePriors' in kwargs.keys():
    kdePriors = kwargs['kdePriors']
  else:
    kdePriors = None

  ##### START VALUES #####

  ###randomly generated start values
  #...these scale factors denote how sigma values are scaled relative to the means
  sigmaScale = 8.5
  phiSigmaScale, sigmaSigmaScale = 8.5, 17.  

  use_thetas = []
  if informedStart == -1:
    print 'Creating values...'
    for i, p in enumerate(ps): #this isn't the most efficient way of doing things, because the phiMu/phiSigma get reassigned every time
      dist = dists[i]
      use_theta = np.zeros((nSubjs[i], nChains, len(p)), dtype = np.float32)
      for param in p:
        if type(starts[param]) == tuple:
          if len(starts[param]) == 2: #tuple of length 2: start values for mu and sigma
            start = starts[param][0]
            sigma = starts[param][1]
            phiSigmaStart = starts[param][1]
            sigmaSigma = phiSigmaStart / sigmaSigmaScale
        else:
          start = starts[param]
          if starts[param] != 0.0: sigma, phiSigmaStart, sigmaSigma = np.abs(float(starts[param]) / sigmaScale), np.abs(float(starts[param]) / phiSigmaScale), np.abs(float(starts[param]) / sigmaSigmaScale)
          elif starts[param] == 0.0: sigma, sigmaSigma = .05, .05

        if np.sum(use_phiSigma[:,hp.index(param)]) == 0: use_phiSigma[:,hp.index(param)] = rtnorm(minSigma, np.inf, phiSigmaStart, sigmaSigma, size = nChains)

        if 'dtnorm' in dist.keys():
          if p.index(param) in dist['dtnorm']:
            if np.sum(use_phiMu[:,hp.index(param)]) == 0: use_phiMu[:,hp.index(param)] = rtnorm(0, np.inf, start, sigma, size = nChains)
            use_theta[:,:,p.index(param)] = rtnorm(0, np.inf, use_phiMu[:,hp.index(param)], use_phiSigma[:,hp.index(param)].reshape(1,nChains) / phiSigmaScale, size = (nSubjs[i], nChains))
        if 'dtnorm01' in dist.keys():
          if p.index(param) in dist['dtnorm01']:
            if np.sum(use_phiMu[:,hp.index(param)]) == 0: use_phiMu[:,hp.index(param)] = rtnorm(0.0, 1.0, start, sigma, size = nChains)
            use_theta[:,:,p.index(param)] = rtnorm(0.0, 1.0, use_phiMu[:,hp.index(param)], use_phiSigma[:,hp.index(param)] / phiSigmaScale, size = (nSubjs[i],nChains))
        if 'beta' in dist.keys():
          if p.index(param) in dist['beta']:
            if np.sum(use_phiMu[:,hp.index(param)]) == 0: use_phiMu[:,hp.index(param)] = rtnorm(0.0, 1.0, start, sigma, size = nChains)
            use_theta[:,:,p.index(param)] = rtnorm(0.0, 1.0, use_phiMu[:,hp.index(param)], use_phiSigma[:,hp.index(param)] / phiSigmaScale, size = (nSubjs[i],nChains))
            #phiSigmaStart = np.abs(float(starts[param])) #beta does not use sigma, but sample size - this tends to be much bigger than the sigma
            #use_phiSigma[:,p.index(param)] = rtnorm(minSigma, np.inf, phiSigmaStart, sigmaSigma, size = nChains)
        if 'norm' in dist.keys():
          if p.index(param) in dist['norm']: 
            if np.sum(use_phiMu[:,hp.index(param)]) == 0: use_phiMu[:,hp.index(param)] = norm.rvs(start, sigma, size = nChains)
            use_theta[:,:,p.index(param)] = norm.rvs(use_phiMu[:,hp.index(param)].reshape(1,nChains), use_phiSigma[:,hp.index(param)].reshape(1,nChains) / phiSigmaScale, size = (nSubjs[i], nChains))
        if 'kde' in dist.keys():
          if p.index(param) in dist['kde']:
            #sample randomly from the KDE for start points
            probMu = kdePriors['mu'][param]['density'] / np.sum(kdePriors['mu'][param]['density'])
            probSigma = kdePriors['sigma'][param]['density'] / np.sum(kdePriors['sigma'][param]['density'])
            use_phiMu[:,hp.index(param)] = [np.random.choice(kdePriors['mu'][param]['support'], p = probMu) for chain in xrange(nChains)]
            use_phiSigma[:,hp.index(param)] = [np.random.choice(kdePriors['sigma'][param]['support'], p = probSigma) for chain in xrange(nChains)]
            use_theta[:,:,p.index(param)] = [np.random.choice(kdePriors['mu'][param]['support'], p = probMu) for chain in xrange(nChains)]

      if 'precision' in dist.keys():
        print 'Using precision scale...'
        #use_phiSigma = 1 / use_phiSigma**2
        use_phiSigma = 1 / use_phiSigma
      if 'logSigma' in dist.keys():
        print 'Sampling log(sigma)...'
        use_phiSigma = np.exp(use_phiSigma)
          
      use_thetas.append(use_theta)

  #use previous fits
  elif informedStart != -1:
    if cont == False:
      print 'Loading start values from a previous fit...'
      use_phiMu = getPhiMu(informedStart)[0:nChains,-1]
      use_phiSigma = getPhiSigma(informedStart)[0:nChains,-1]
      n = 0
      use_thetas = range(len(ps))
      for i in xrange(len(ps)):
        p = ps[i]
        use_theta = np.zeros((nSubjs[i], nChains, len(p)), dtype = np.float32)
        for subj in xrange(nSubjs[i]): use_theta[subj] = getSubj(subj+n,informedStart)[0:nChains,-1]
        n += nSubjs[i]
        use_thetas[i] = use_theta

    elif cont == True:
      print 'Continuing a previous fit...'
      thetas = []
      n = 0
      for i in xrange(len(ps)):
        theta0 = getSubj(0, informedStart)[0:nChains]
        nChains, nPrevious, nParams = theta0.shape
        theta = np.zeros((nSubjs[i],nChains,nPrevious,nParams))
        theta[0] = theta0
        for subj in xrange(1,nSubjs[i]): theta[subj] = getSubj(subj + n,informedStart)[0:nChains]
        n += nSubjs[i]
        theta = np.concatenate((theta, np.zeros((nSubjs[i],nChains,nmc,nParams))), axis = 2)

        theta.append(thetas)
        use_theta = theta[:,:,nPrevious-1]
        use_thetas.append(use_theta)

      phiMu, phiSigma = getPhiMu(informedStart)[0:nChains], getPhiSigma(informedStart)[0:nChains]
      use_phiMu, use_phiSigma = phiMu[:,nPrevious-1], phiSigma[:,nPrevious-1]

      phiMu = np.concatenate((phiMu, np.zeros((nChains,nmc,nParams))), axis = 1)
      phiSigma = np.concatenate((phiSigma, np.zeros((nChains,nmc,nParams))), axis = 1)
      like = np.concatenate((getLike(informedStart)[:,0:nChains], np.zeros((nSubj,nChains,nmc))), axis = 2)
      weight = np.concatenate((getWeight(informedStart)[:,0:nChains], np.zeros((nSubj,nChains,nmc))), axis = 2)
      hyperWeight = np.concatenate((getHyperWeight(informedStart)[0:nChains], np.zeros((nChains,nmc,nParams))), axis = 1)
      
      use_weight, useHyperWeight = weight[:,:,nPrevious-1], hyperWeight[:,nPrevious-1]

      del theta0
      gc.collect()

  print 'Start values generated...'
 
  ### generate the initial set of likelihoods
  #...have to do this a little differently here, create a list containing all the theta parameters
  flattened = [list(use_thetas[i]) for i in xrange(len(use_thetas))]
  use_theta_subjs = []
  for i in xrange(len(flattened)): use_theta_subjs += flattened[i]
  
  indices = []
  for i in xrange(len(nSubjs)): indices += [i] * nSubjs[i]
  nSubj = len(np.unique(data['subj']))
  subj_ps = [ps[indices[subj]] for subj in xrange(nSubj)] #this list contains nSubj parameter lists, this allows for one call to Parallel instead of one for each group below

  use_like = np.array(Parallel(n_jobs = nJobs)(delayed(log_dens_like)(use_theta_subjs[subj], data[data['subj'] == subject], subj_ps[subj]) for subj, subject in enumerate(np.unique(data['subj']))))
  prior = log_dens_hyperBetween(use_thetas, use_phiMu, use_phiSigma, dists, distHiers, nSubj)
  use_weight = prior + use_like

  ### parameter check ###
  chains = range(nChains)
  use_hyperWeight = log_dens_hyper_and_priorBetween(use_thetas, use_phiMu, use_phiSigma, priors, dists, distHiers, distHierAll, hp, chains, kdePriors)
  if (np.mean(np.isfinite(np.mean(use_weight))) == 1.0) & (np.mean(np.isfinite(use_hyperWeight)) == 1.0):
    print 'Check passed!'

  else:
    print 'Check failed...'
    if np.mean(np.isfinite(np.mean(use_like))) != 1.0:
      print 'Bad likelihood values...'
      print np.mean(np.isfinite(use_like))
    if np.mean(np.isfinite(prior)) != 1.0:
      print 'Subject parameters illegal under hyperparameters...' 
    if (np.mean(np.isfinite(use_hyperWeight)) != 1.0):
      print 'Hyperparameters illegal under priors...'
      
      for param in p:
        if np.mean(np.isfinite(use_hyperWeight[:,p.index(param)])) != 1.0:
          print param
      print np.mean(np.isfinite(use_hyperWeight))

  #replace nans with negative inf
  use_weight[np.isnan(use_weight)] = -1 * np.inf
  use_like[np.isnan(use_like)] = -1 * np.inf
 
  print 'Beginning sampling...'
  if 'migrationStart' in kwargs.keys(): migrationStart = kwargs['migrationStart']
  else: migrationStart = 15000
  if 'migrationStop' in kwargs.keys(): migrationStop = kwargs['migrationStop']
  else: migrationStop = 10**1000
  gc.collect()

  #create structures right away if burnin is zero
  if (burnin == 0) & (cont == False):
    print 'Burnin period over... recording samples...'
    thetas = []
    for g in xrange(3):
      nSubj, nParams, use_theta = nSubjs[g], len(ps[g]), use_thetas[g]
      theta = np.zeros((nSubj, nChains, nmc, nParams), dtype = np.float32)
      theta[:,:,0] = use_theta
      thetas.append(theta)
    nSubj, nParams = len(np.unique(data['subj'])), len(hp)
    phiMu, phiSigma = np.zeros((nChains, nmc, nParams), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32)
    hyperWeight, like, weight = np.zeros((nChains, nmc, nParams), dtype = np.float32), np.zeros((nSubj, nChains, nmc), dtype = np.float32), np.zeros((nSubj, nChains, nmc))   
    phiMu[:,0], phiSigma[:,0], hyperWeight[:,0], like[:,:,0], weight[:,:,0] = use_phiMu, use_phiSigma, use_hyperWeight, use_like, use_weight
    i = 0
     
  ### MCMC loop ###
  MCMC = True
  step = 0
  
  if cont == True:
    step = nPrevious
    i = nPrevious #ensure that MCMC starts where it left off
    nmc += nPrevious
  print str(step)

  if 'randomGamma' in kwargs.keys(): randomGamma = kwargs['randomGamma']
  else: randomGamma = False 

  if type(rp) == np.float: rp = np.array([rp])
  rp1 = rp

  while MCMC == True:

    step += 1

    #recalculate likelihoods (for the PDA route)
    recalc = False
    if recalculate != False:
      #recalculate less frequently during burn-in
      if step < burnin:
        if step % (recalculate * 2) == 0: recalc = True
      else:
        if step % recalculate == 0: recalc = True

    #gamma parameters
    if ((gamma1 == True) | (step < burnin)) & ((step + 5) % 10 == 0):
      gammaProps = [.98] * len(ps)
      if recalculate == True: recalc = True
    elif (gammat0 == True) & (step % 10 == 0):
      t = [p.index(param) for param in p if ('t0' in param) | ('st0' in param)] #just upscale the problem parameters
      if gammaUnif == False: gammaProp = np.ones((1,1,nParams)) * (2.38 / np.sqrt(2 * nParams))
      elif gammaUnif == True: gammaProp = np.ones((nSubj,nChains,nParams)) * np.random.uniform(.5, 1.0, size = (nSubj,nChains,1))
      gammaProp[:,:,t] = 1.1
    elif (gammaVaa != False) & (step % 10 == 0):
      #source memory specific one
      gammaProps = []
      for g in xrange(len(ps)):
        p = ps[g]
        nParams = len(p)
        t = [p.index(param) for param in p if ('logVaa' in param) | ('logVab' in param)] #just upscale the problem parameters
        if gammaUnif == True: gammaProp = np.ones((nChains,nParams)) * np.random.uniform(.5, 1.0, size = (nChains,1))
        else: gammaProp = np.ones((1,nParams)) * (2.38 / np.sqrt(2 * nParams))
        if gammat0 == True: gammaVal = 1.1
        else: gammaVal = kwargs['gammaVaa']
        gammaProp[:,t] = gammaVal
        gammaProps.append(gammaProp)
    elif (gammar != False) & ((step + 3) % 10 == 0):
      #source memory specific one
      gammaProps = []
      for g in xrange(len(ps)):
        p = ps[g]
        nParams = len(p)
        t = [p.index(param) for param in p if ('logr' in param)] #just upscale the problem parameters
        if gammaUnif == True: gammaProp = np.ones((nChains,nParams)) * np.random.uniform(.5, 1.0, size = (nChains,1))
        else: gammaProp = np.ones((1,nParams)) * (2.38 / np.sqrt(2 * nParams))
        if gammat0 == True: gammaVal = 1.1
        else: gammaVal = kwargs['gammar']
        gammaProp[:,t] = gammaVal
        gammaProps.append(gammaProp)
    elif (gammabgNoise != False) & ((step + 8) % 20 == 0):
      #source memory specific one
      gammaProps = []
      for g in xrange(len(ps)):
        p = ps[g]
        nParams = len(p)
        t = [p.index(param) for param in p if ('logbgNoise' in param)] #just upscale the problem parameters
        if gammaUnif == True: gammaProp = np.ones((nChains,nParams)) * np.random.uniform(.5, 1.0, size = (nChains,1))
        else: gammaProp = np.ones((1,nParams)) * (2.38 / np.sqrt(2 * nParams))
        if gammat0 == True: gammaVal = 1.1
        else: gammaVal = kwargs['gammabgNoise']
        gammaProp[:,t] = gammaVal
        gammaProps.append(gammaProp)

    elif gammaUnif == True: gammaProps = [np.ones((nSubj,nChains,len(p))) * np.random.uniform(.5, 1.0, size = (nSubj,nChains,1)) for p in ps]
    elif gammaUnif == False:  gammaProps = [2.38 / np.sqrt(2 * nParams)] * len(ps)
    #gammaProp = np.random.uniform(.5, 1)
    if (step < burnin) | (step % 15 == 0): rp = rp1 * 5
    else: rp = rp1

    ### create the data structures if burnin period is over
    if step == burnin:
      print 'Burnin period over... recording samples...'
      thetas = []
      for g in xrange(3):
        nSubj, nParams = nSubjs[g], len(ps[g])
        theta = np.zeros((nSubj, nChains, nmc, nParams), dtype = np.float32)
        thetas.append(theta)
      nSubj, nParams = len(np.unique(data['subj'])), len(hp)
      phiMu, phiSigma = np.zeros((nChains, nmc, nParams), dtype = np.float32), np.zeros((nChains, nmc, nParams), dtype = np.float32)
      hyperWeight, like, weight = np.zeros((nChains, nmc, nParams), dtype = np.float32), np.zeros((nSubj, nChains, nmc), dtype = np.float32), np.zeros((nSubj, nChains, nmc))   
      i = 0

    ### crossover hyper step: sample new group parameters ###
    chains = range(nChains)
    shuffle(chains)
    
    use_phiMu, use_phiSigma, use_hyperWeight = crossoverHyperBetween(use_thetas, use_phiMu, use_phiSigma, priors, dists, distHiers, distHierAll, hp, rp, chains, kdePriors) #chains can't be easily applied to theta here, so it's passed in

    ### crossover step: sample new subject parameters ###
    chains = range(nChains)
    shuffle(chains)

    use_thetas, use_like, use_weight = crossoverBetween(use_like, use_thetas, use_phiMu[chains], use_phiSigma[chains], nSubjs, data, log_dens_like, ps, dists, distHiers, rp, gammaProps, nJobs, randomGamma = randomGamma, recalculate = recalc)

    ### migration step (optional) ###
    if 'migration' in kwargs.keys():
      if step == migrationStart: print 'Migration beginning!'
      elif step == migrationStop: print 'Migration ceasing!'

      #run hypermigration less frequently
      if (step > migrationStart) & (step < migrationStop) & (step % (kwargs['migration'] * 1) == 0):

        use_phiMu, use_phiSigma, use_hyperWeight = migrationHyperBetween(use_thetas, use_phiMu, use_phiSigma, priors, distHierAll, ps, hp, rp * 10, kdePriors = None)

      #subject migration
      if (step > migrationStart) & (step < migrationStop) & (step % kwargs['migration'] == 0):
        lower, x = 0, 0
        #use the same migration function, but just loop separately through the groups. It's slower, but far easier
        for g in xrange(len(use_thetas)):
          chains = range(nChains)
          shuffle(chains)

          use_theta, p, nSubj, dist = use_thetas[g], ps[g], nSubjs[g], dists[g]
          p2 = [hp.index(param) for param in p] #for indexing hypers
          upper = lower + nSubj

          subjectList = np.unique(data['subj'])[lower:upper]

          #below is another case where Python can get screwed up if you use multiple lists to index different dimensions of an array
          newVals = Parallel(n_jobs = -1)(delayed(migration)(use_theta[subj], use_phiMu[chains][:,p2], use_phiSigma[chains][:,p2], use_like[subj+x], data[data['subj'] == subject], log_dens_like, p, dist, rp * 10) for subj, subject in enumerate(subjectList))
          for subj in xrange(nSubj):
            use_theta[subj], use_like[subj+x], use_weight[subj+x] = newVals[subj]

          lower = upper
          x += nSubj
    
    #update the data structures
    if (step >= burnin) & (step % thin == 0):
      phiMu[:,i], phiSigma[:,i], hyperWeight[:,i], like[:,:,i], weight[:,:,i]  = use_phiMu, use_phiSigma, use_hyperWeight, use_like, use_weight
      for g in xrange(len(thetas)): #loop through the groups for assignment
        thetas[g][:,:,i] = use_thetas[g]
        
      i += 1
      if i == nmc: MCMC = False

    #print progress
    if (step + 1) % 25 == 0:
      print '*',
    if (step + 1) % 500 == 0:
      gc.collect()
      print step + 1

  print 'Finished!'
  #/end of MCMC loop
  return thetas, phiMu, phiSigma, like, weight, hyperWeight, priors

def calculateWAIC(data, params, log_dens_like, theta, thin = 10, nJobs = -1):

  nSubj, nChains, nSamples = len(theta), theta[0].shape[0], theta[0].shape[1]
  nSamples = nSamples / thin
  
  #likes: chains x samples x data points: if prior/hypers were to be considered, would be extended by 1 (data points + 1)
  likes = [np.zeros((theta[subj].shape[0],nSamples,len(data[data['subj'] == subject]))) for subj, subject in enumerate(np.unique(data['subj']))] #list form, because each subject has different n data points

  #recalculate likelihoods with full = True to get likelihoods for each data point
  print 'Calculating likelihoods...'
  for i in xrange(nSamples):

    ls = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj][:,i*thin], data[data['subj'] == subject], params, full = True) for subj, subject in enumerate(np.unique(data['subj'])))

    for j, l in enumerate(ls):
      likes[j][:,i] = l
    if (i + 1) % 25 == 0: print '*',

  print 'Likelihoods collected... Calculating WAIC'
  pwaic, lpd = np.zeros(nSubj), np.zeros(nSubj)
  for subj, subject in enumerate(np.unique(data['subj'])):
    like = likes[subj]
    nData = len(data[data['subj'] == subject])
    like = like.reshape((like.shape[0] * like.shape[1], nData))
    pwaic[subj] = np.nansum(np.nanvar(np.log(like), axis = 0)) #take the variance of LOG likelihood over SAMPLES, then sum over DATA points
    lpd[subj] = np.nansum(np.log(np.nanmean(like, axis = 0))) #average likelihood over SAMPLES, then sum over LOG likelihood over DATA points
    #like = np.log(like.reshape((nSamples * nChains, nData)))
    #pwaic[subj] = np.sum(np.var(like, axis = 0)) #take the variance of LOG likelihood over SAMPLES, then sum over DATA points
    #lpd[subj] = np.sum(np.sum(like, axis = 0) - np.log(nSamples * nChains))
  
  elpd_waic = lpd - pwaic
  waic = -2 * elpd_waic

  fields = [('lpd', np.float32), ('pwaic', np.float32), ('waic', np.float32)]
  complete = np.zeros(nSubj, fields)
  complete['lpd'] = tuple(lpd)
  complete['pwaic'] = tuple(pwaic)
  complete['waic'] = tuple(waic)

  return np.sum(lpd), np.sum(pwaic), np.sum(waic), complete

def calculateWAIC_all(data, params, log_dens_like, theta, thin = 10, nJobs = -1):

  nSubj, nChains, nSamples = theta.shape[0], theta.shape[1], theta.shape[2]
  pwaic, lpd = np.zeros(nSubj), np.zeros(nSubj)

  #recalculate likelihoods with full = True to get likelihoods for each data point
  print 'Calculating likelihoods...'
  for subj, subject in enumerate(np.unique(data['subj'])):
    nData = len(data[data['subj'] == subject])
    like = np.zeros((nChains,nSamples,nData))
    for i in xrange(0, nSamples, 100):
      ls = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,:,i+j], data[data['subj'] == subject], params, full = True) for j in xrange(100))
      for j in xrange(100): like[:,i+j] = ls[j]
    like = like.reshape((nSamples * nChains, nData))
    pwaic[subj] = np.sum(np.var(np.log(like), axis = 0)) #take the variance of LOG likelihood over SAMPLES, then sum over DATA points
    lpd[subj] = np.sum(np.log(np.mean(like, axis = 0))) #average likelihood over SAMPLES, then sum over LOG likelihood over DATA points
    del like
    print subj,
  
  elpd_waic = lpd - pwaic
  waic = -2 * elpd_waic

  fields = [('lpd', np.float32), ('pwaic', np.float32), ('waic', np.float32)]
  complete = np.zeros(nSubj, fields)
  complete['lpd'] = tuple(lpd)
  complete['pwaic'] = tuple(pwaic)
  complete['waic'] = tuple(waic)

  return np.sum(lpd), np.sum(pwaic), np.sum(waic), complete

def calculateWAIC_prior(data, params, dist, log_dens_like, theta, phiMu, phiSigma, thin = 10, nJobs = -1):

  nSubj, nChains, nSamples, nParams = theta.shape
  nSamples = nSamples / thin
  
  #likes: chains x samples x data points + 1: extra 1 is for hypers
  likes = [np.zeros((nChains,nSamples,len(data[data['subj'] == subject]) + nParams)) for subj, subject in enumerate(np.unique(data['subj']))] #list form, because each subject has different n data points

  #recalculate likelihoods with full = True to get likelihoods for each data point
  print 'Calculating likelihoods...'
  for i in xrange(nSamples):

    ls = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,:,i*thin], data[data['subj'] == subject], params, full = True) for subj, subject in enumerate(np.unique(data['subj'])))
    prior = np.exp(log_dens_hyper_separate(theta[:,:,i*thin], phiMu[:,i*thin], phiSigma[:,i*thin], dist)) #get prior density for all subjects

    for j, l in enumerate(ls):
      nData = len(data[data['subj'] == np.unique(data['subj'])[j]])
      likes[j][:,i,0:nData] = l
      likes[j][:,i,nData:nData + nParams] = prior[j] #assign prior density for subject j in the last cell
    if (i + 1) % 25 == 0: print '*',

  print 'Likelihoods collected... Calculating WAIC'
  pwaic, lpd = np.zeros(nSubj), np.zeros(nSubj)
  for subj, subject in enumerate(np.unique(data['subj'])):
    like = likes[subj]
    nData = len(data[data['subj'] == subject])
    like = like.reshape((nSamples * nChains, nData + nParams))
    pwaic[subj] = np.sum(np.var(np.log(like), axis = 0)) #take the variance of LOG likelihood over SAMPLES, then sum over DATA points
    lpd[subj] = np.sum(np.log(np.mean(like, axis = 0))) #average likelihood over SAMPLES, then sum over LOG likelihood over DATA points
    print subj, lpd[subj], pwaic[subj]

  elpd_waic = lpd - pwaic
  waic = -2 * elpd_waic

  fields = [('lpd', np.float32), ('pwaic', np.float32), ('waic', np.float32)]
  complete = np.zeros(nSubj, fields)
  complete['lpd'] = tuple(lpd)
  complete['pwaic'] = tuple(pwaic)
  complete['waic'] = tuple(waic)

  return np.sum(lpd), np.sum(pwaic), np.sum(waic), complete

#this is mainly for the SDT fits, where the n data points != the dimensionality that should be used
def calculateWAIC2(data, params, log_dens_like, theta, thin = 10, nJobs = -1):

  nSubj, nChains, nSamples = theta.shape[0], theta.shape[1], theta.shape[2]
  nSamples = nSamples / thin
  
  #likes: chains x samples x data points: if prior/hypers were to be considered, would be extended by 1 (data points + 1)

  #get the likelihoods first
  ls = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,:,0], data[data['subj'] == subject], params, full = True) for subj, subject in enumerate(np.unique(data['subj'])))

  #use the likelihoods in ls to determine the shapes of each array, then insert ls into likes
  likes = [np.zeros((nChains,nSamples,ls[subj].shape[1])) for subj, subject in enumerate(np.unique(data['subj']))] #list form, because each subject has different n data points
  for j, l in enumerate(ls):
    likes[j][:,0] = l

  #recalculate likelihoods with full = True to get likelihoods for each data point
  print 'Calculating likelihoods...'
  for i in xrange(1,nSamples):

    ls = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,:,i*thin], data[data['subj'] == subject], params, full = True) for subj, subject in enumerate(np.unique(data['subj'])))

    for j, l in enumerate(ls):
      likes[j][:,i] = l
    if (i + 1) % 25 == 0: print '*',

  print 'Likelihoods collected... Calculating WAIC'
  pwaic, lpd = np.zeros(nSubj), np.zeros(nSubj)
  for subj, subject in enumerate(np.unique(data['subj'])):
    like = likes[subj]
    nData = like.shape[2]
    like = like.reshape((nSamples * nChains, nData))
    pwaic[subj] = np.sum(np.var(np.log(like), axis = 0)) #take the variance of LOG likelihood over SAMPLES, then sum over DATA points
    lpd[subj] = np.sum(np.log(np.mean(like, axis = 0))) #average likelihood over SAMPLES, then sum over LOG likelihood over DATA points
  
  elpd_waic = lpd - pwaic
  waic = -2 * elpd_waic

  fields = [('lpd', np.float32), ('pwaic', np.float32), ('waic', np.float32)]
  complete = np.zeros(nSubj, fields)
  complete['lpd'] = tuple(lpd)
  complete['pwaic'] = tuple(pwaic)
  complete['waic'] = tuple(waic)

  return np.sum(lpd), np.sum(pwaic), np.sum(waic), complete

def calculateWAIC2Between(data, params, log_dens_like, thetas, ps, nSubjs, thin = 10, nJobs = -1):

  nChains, nSamples = thetas[0].shape[1], thetas[0].shape[2]
  nSubj = len(np.unique(data['subj']))
  nSamples = nSamples / thin
  
  #likes: chains x samples x data points: if prior/hypers were to be considered, would be extended by 1 (data points + 1)

  #set up the thetas and ps
  theta = range(nSubj)
  params = range(nSubj)
  subj = 0
  for g in xrange(len(thetas)):
    for i in xrange(thetas[g].shape[0]):
      theta[subj] = thetas[g][i]
      params[subj] = ps[g]
      subj += 1

  ls = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj][:,0], data[data['subj'] == subject], params[subj], full = True) for subj, subject in enumerate(np.unique(data['subj'])))

  #use the likelihoods in ls to determine the shapes of each array, then insert ls into likes
  likes = [np.zeros((nChains,nSamples,ls[subj].shape[1])) for subj, subject in enumerate(np.unique(data['subj']))] #list form, because each subject has different n data points
  for j, l in enumerate(ls):
    likes[j][:,0] = l

  #recalculate likelihoods with full = True to get likelihoods for each data point
  print 'Calculating likelihoods...'
  for i in xrange(1,nSamples):

    ls = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj][:,i*thin], data[data['subj'] == subject], params[subj], full = True) for subj, subject in enumerate(np.unique(data['subj'])))

    for j, l in enumerate(ls):
      likes[j][:,i] = l
    if (i + 1) % 25 == 0: print '*',

  print 'Likelihoods collected... Calculating WAIC'
  pwaic, lpd = np.zeros(nSubj), np.zeros(nSubj)
  for subj, subject in enumerate(np.unique(data['subj'])):
    like = likes[subj]
    nData = like.shape[2]
    like = like.reshape((nSamples * nChains, nData))
    pwaic[subj] = np.sum(np.var(np.log(like), axis = 0)) #take the variance of LOG likelihood over SAMPLES, then sum over DATA points
    lpd[subj] = np.sum(np.log(np.mean(like, axis = 0))) #average likelihood over SAMPLES, then sum over LOG likelihood over DATA points
  
  elpd_waic = lpd - pwaic
  waic = -2 * elpd_waic

  fields = [('lpd', np.float32), ('pwaic', np.float32), ('waic', np.float32)]
  complete = np.zeros(nSubj, fields)
  complete['lpd'] = tuple(lpd)
  complete['pwaic'] = tuple(pwaic)
  complete['waic'] = tuple(waic)

  return np.sum(lpd), np.sum(pwaic), np.sum(waic), complete

def calculateDIC_hybrid(data, params, hierParams, groupParams, dist, distHier, distGroup, priorsGroup, log_dens_like, theta, phiMu, phiSigma, weight, op = np.mean, burnin = 0):

  nSubj, nChains, nSamples, nParams = theta.shape
  nSubj = len(np.unique(data['subj']))

  sumWeight = np.sum(weight[:,:,burnin:], axis = 0) #sum over subjects
  priorWeight = log_dens_prior2(phiMu[:,:,groupParams], priorsGroup, distGroup).reshape(nChains,nSamples) #include the likelihoods of the group parameters
  sumWeight += priorWeight
  meanFullWeight = op(sumWeight)
  theta, phiMu, phiSigma = theta[:,:,burnin:], phiMu[:,burnin:], phiSigma[:,burnin:]
  #average over chains and samples, not over subjects
  if op == np.mean:
    meanPhiMu = op(phiMu, axis = (0, 1))
    meanPhiSigma = op(phiSigma, axis = (0, 1))
    meanTheta = op(theta, axis = (1,2))
  elif op == np.median:
    meanPhiMu = np.array([np.median(phiMu[:,:,prm]) for prm in xrange(len(params))])
    meanPhiSigma = np.array([np.median(phiSigma[:,:,prm]) for prm in xrange(len(params))])
    meanTheta = np.array([[np.median(theta[subj,:,:,prm]) for prm in xrange(len(params))] for subj in xrange(theta.shape[0])])

  meanThetaHier, meanPhiMuGroup, meanPhiMuHier, meanPhiSigmaHier = meanTheta[:,hierParams], meanPhiMu[groupParams], meanPhiMu[hierParams], meanPhiSigma[hierParams]
  meanPrior = 0
  if 'dtnorm' in distHier.keys():
    meanPrior += np.sum(np.log(dtnorm(meanThetaHier[:,distHier['dtnorm']], 0, np.inf, loc = meanPhiMuHier[distHier['dtnorm']], scale = meanPhiSigmaHier[distHier['dtnorm']])))
  if 'dtnorm01' in distHier.keys():
    meanPrior += np.sum(np.log(dtnorm(meanThetaHier[:,distHier['dtnorm01']], 0, 1.0, loc = meanPhiMuHier[distHier['dtnorm01']], scale = meanPhiSigmaHier[distHier['dtnorm01']])))
  if 'norm' in distHier.keys():
    meanPrior += np.sum(np.log(norm.pdf(meanThetaHier[:,distHier['norm']], loc = meanPhiMuHier[distHier['norm']], scale = meanPhiSigmaHier[distHier['norm']])))

  priorMus, priorSigmas = priorsGroup
  if 'dtnorm' in distGroup.keys():
    meanPrior += np.sum(np.log(dtnorm(meanPhiMuGroup[distGroup['dtnorm']], 0, np.inf, loc = priorMus[distGroup['dtnorm']], scale = priorSigmas[distGroup['dtnorm']])))
  if 'dtnorm01' in distGroup.keys():
    meanPrior += np.sum(np.log(dtnorm(meanPhiMuGroup[distGroup['dtnorm01']], 0, 1.0, loc = priorMus[distGroup['dtnorm01']], scale = priorSigmas[distGroup['dtnorm01']])))
  if 'norm' in distGroup.keys():
    meanPrior += np.sum(np.log(norm.pdf(meanPhiMuGroup[distGroup['norm']], loc = priorMus[distGroup['norm']], scale = priorSigmas[distGroup['norm']])))

  weightAtMean = np.sum(np.array([log_dens_like(meanTheta[subj], data[data['subj'] == subject], params) for subj, subject in enumerate(np.unique(data['subj']))])) + meanPrior
  
  D = -2 * meanFullWeight
  pD = D + (2 * weightAtMean)
  DIC = D + pD
  return D, pD, DIC  


def calculateDICSubject(data, params, dist, log_dens_like, theta, weight, priors, burnin = 2000):
  weight = np.mean(weight, axis = 1) #take the mean with respect to chains
  inc = weight > np.mean(weight) - (2 * np.std(weight)) #cut anything more than two SDs below the mean
  theta, weight = theta[inc,burnin:], weight[inc]
  meanFullWeight = np.mean(weight)
  #average over chains and samples, not over subjects
  meanTheta = np.mean(theta, axis = (0, 1))

  weightAtMean = log_dens_like(meanTheta, data, params)
  weightAtMean += np.sum(np.log(lognorm.pdf(meanTheta, scale = priors[0], s = priors[1])))
  weightAtMean = np.sum(weightAtMean)
  
  D = -2 * meanFullWeight
  pD = D + (2 * weightAtMean)
  DIC = D + pD
  return D, pD, DIC

#### GENERATE POSTERIOR PREDICTIVES #####

def synthData(data, p, log_dens_like, theta, burnin = 1000, thinBy = 40, nJobs = -1, factors = [], lower = .2, upper = 2.5, zROC = False, doubleFactor = False, ttype = 'type', conf = False, x = 1):
  nSubj,nChains,nmc,nParams = theta.shape
  print p
  theta = theta[:,:,burnin:]
  theta = theta[:,:,np.arange(0,nmc-burnin,thinBy)]
  nSims = theta.shape[2]
  if x == 1:
    subjs = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,:,0], data[data['subj'] == subject], p, simulate = True) for subj, subject in enumerate(np.unique(data['subj'])))
  else:
    subjs = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,:,0], data[data['subj'] == subject], p, simulate = True, x = x) for subj, subject in enumerate(np.unique(data['subj'])))

  #have to keep separate arrays for each subject due to size issues
  #...what's stored is a record array with fields 'corr' and 'RT'
  #...shape of the array is number of simulations, number of chains, and number of responses in the task
  subjArrays = []
  if conf == False: fields = [('corr', np.uint8), ('RT', np.float32)]
  else: fields = [('corr', np.uint8), ('RT', np.float32), ('conf', np.int8)]

  for s, subj in enumerate(subjs):
    r = subj
    nResps = r[1].shape[1]
    if x == 1: subjArray = np.zeros((nSims,nChains,nResps), fields)
    else: subjArray = np.zeros((nSims,nChains,nResps,x), fields)
    subjArray[0]['RT'], subjArray[0]['corr'] = r[0], r[1]
    if conf == True: subjArray[0]['conf'] = r[2]
    subjArrays.append(subjArray)

  for sim in np.arange(1,nSims):
    if x == 1: subjs = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,:,sim], data[data['subj'] == subject], p, simulate = True) for subj, subject in enumerate(np.unique(data['subj'])))
    else: subjs = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,:,sim], data[data['subj'] == subject], p, simulate = True, x = x) for subj, subject in enumerate(np.unique(data['subj'])))
    for s, subj in enumerate(subjs):
      subjArrays[s][sim]['RT'], subjArrays[s][sim]['corr'] = subj[0], subj[1]
      if conf == True: subjArrays[s][sim]['conf'] = subj[2]
    if (sim + 1) % 25 == 0: print '*',

  #reshape the arrays so that chains and samples collapse into each other
  for s, subj in enumerate(subjs): subjArrays[s] = subjArrays[s].reshape(-1, subjArrays[s].shape[-1])

  #save quantile subj averages
  qs = [.1, .5, .9]
  summaries = {}
  summaries['oldnew_qs=' + str(qs)] = synthSubjAverage_noFactor(data, subjArrays, qs, lower = lower, upper = upper, zROC = zROC, ttype = ttype)
  if conf == False:
    fnSingle, fnDouble = synthSubjAverage_singleFactor, synthSubjAverage_doubleFactor
  else:
    fnSingle, fnDouble = synthSubjAverage_singleFactorConf, synthSubjAverage_doubleFactorConf
  for factor in factors:
    qs = [.1, .5, .9]
    print factor, qs
    summaries[factor + '_qs=' + str(qs)] = fnSingle(data, subjArrays, factor, qs = qs, lower = lower, upper = upper, zROC = zROC)

  if doubleFactor != False:
    qs = [.1, .5, .9]
    for dF in doubleFactor:
      summaries[dF[0] + dF[1] + '_qs=' + str(qs)] = fnDouble(data, subjArrays, factor1 = dF[0], factor2 = dF[1], qs = qs, lower = lower, upper = upper)

  return subjArrays, summaries

def synthSubjAverage_noFactor(data, model, qs = [.1, .5, .9], lower = .2, upper = 2.5, zROC = False, ttype = 'type', x = 1):
  print 'Subject average no factor'
  nSamples = model[0].shape[0]
  nSubj = len(np.unique(data['subj']))
  nOldnew = len(np.unique(data[ttype]))
  ratesModel = np.zeros((nOldnew,nSamples))
  if zROC == True:
    zRatesModel = np.zeros((nOldnew,nSamples))
    if 'correct' in data.dtype.fields: corr_field = 'correct'
    elif 'corr' in data.dtype.fields: corr_field = 'corr'
  modelCorrRTs = np.zeros((nOldnew,len(qs),nSamples))
  modelErrorRTs = np.zeros((nOldnew,len(qs),nSamples))

  if 'Target' in data[ttype]: oldnews = ['Target', 'Lure']
  elif 'target' in data[ttype]: oldnews = ['target', 'lure']
  elif 'left' in data[ttype]: oldnews = ['left', 'right']
  elif 'strong' in data[ttype]: oldnews = ['lure', 'weak', 'strong']
  #elif 'lure1' in data[ttype]: oldnews = ['elf1', 'elf2', 'lure1', 'lure2', 'target1', 'target2', 'target3']
  elif 'lure2' in data[ttype]: oldnews = ['elf1dim1', 'elf1dim2', 'elf1dim3', 'lure1', 'elf2dim1', 'elf2dim2', 'elf2dim3', 'lure2', 'target1', 'target2', 'target3']
  else: oldnews = np.unique(data[ttype])

  print nOldnew, len(oldnews)

  if 'response' in data.dtype.names: resp = 'response'
  elif 'resp' in data.dtype.names: resp = 'resp'

  for o, oldnew in enumerate(oldnews):
    subjs = np.array([np.mean(model[sbj][:,np.where((data[data['subj'] == subject][ttype] == oldnew))[0]]['corr'], axis = 1) for sbj, subject in enumerate(np.unique(data['subj']))])
    #reverse for lure items (get the FAR)
    if oldnew in ['lure', 'Lure', 'elf1', 'elf2', 'elf1dim1', 'elf1dim2', 'elf1dim3', 'elf2dim1', 'elf2dim2', 'elf2dim3', 'lure1', 'lure2']: subjs = 1.0 - subjs
    rate = np.mean(subjs, axis = 0)
    #edge corrections
    if zROC == True:
      total = np.array([len(data[(data['subj'] == subject) & (data[ttype] == oldnew)][corr_field]) for sbj, subject in enumerate(np.unique(data['subj']))])
      total = total.reshape((nSubj, 1))
      transform = ((subjs * total) + .5) / (total + 1) 
      zrate = np.mean(norm.ppf(transform), axis = 0) 
  
    RTs = np.zeros((nSubj,len(qs),nSamples))

    #loop over subjects and samples to get quantiles for each sample
    for sbj,subject in enumerate(np.unique(data['subj'])):
      for sample in xrange(nSamples):
        RTs[sbj,:,sample] = mstats.mquantiles(model[sbj][sample,(data[data['subj'] == subject][ttype] == oldnew) &
                                                    (model[sbj][sample]['RT'] > lower) &
                                                    (model[sbj][sample]['RT'] < upper) & 
                                                    (model[sbj][sample]['corr'] == 1)]['RT'], prob = qs)

    modelCorrRTs[o] = np.mean(RTs, axis = 0) #average over subjects, preserve quantiles and samples

    RTs = np.zeros((nSubj,len(qs),nSamples))
    for sbj,subject in enumerate(np.unique(data['subj'])):
      for sample in xrange(nSamples):
        RTs[sbj,:,sample] = mstats.mquantiles(model[sbj][sample,(data[data['subj'] == subject][ttype] == oldnew) &
                                                    (model[sbj][sample]['RT'] > lower) &
                                                    (model[sbj][sample]['RT'] < upper) & 
                                                    (model[sbj][sample]['corr'] == 0)]['RT'], prob = qs)
    modelErrorRTs[o] = np.mean(RTs, axis = 0)
    ratesModel[o] = rate
    if zROC == True: zRatesModel[o,f] = zrate

  if np.min(modelErrorRTs < lower): print 'Problem: lower'
  if np.max(modelErrorRTs > upper): print 'Problem: upper'

  if zROC == False: return ratesModel, modelCorrRTs, modelErrorRTs
  elif zROC == True: return ratesModel, zRatesModel, modelCorrRTs, modelErrorRTs 

#generate subject average for a single factor (WF, speed/accuracy, etc.)
#...zROC argument: z transform before subject average
def synthSubjAverage_singleFactor(data, model, factor, qs = [.1, .5, .9], lower = .2, upper = 2.5, zROC = False):
  print 'Subject average single factor: ' + str(factor)
  nSamples = model[0].shape[0]
  nSubj = len(np.unique(data['subj']))
  nLevels = len(np.unique(data[factor]))
  nOldnew = len(np.unique(data['type']))
  ratesModel = np.zeros((nOldnew,nLevels,nSamples))
  if zROC == True:
    zRatesModel = np.zeros((nOldnew,nLevels,nSamples))
    if 'correct' in data.dtype.fields: corr_field = 'correct'
    elif 'corr' in data.dtype.fields: corr_field = 'corr'
  modelCorrRTs = np.zeros((nOldnew,nLevels,len(qs),nSamples))
  modelErrorRTs = np.zeros((nOldnew,nLevels,len(qs),nSamples))

  if 'Target' in data['type']: oldnews = ['Target', 'Lure']
  elif 'target' in data['type']: oldnews = ['target', 'lure']
  elif 'left' in data['type']: oldnews = ['left', 'right']
  elif 'strong' in data['type']: oldnews = ['lure', 'weak', 'strong']
  else: oldnews = np.unique(data['type'])

  if 'response' in data.dtype.names: resp = 'response'
  elif 'resp' in data.dtype.names: resp = 'resp'
 
  #having predefined label orders ensures that there is no confusion about their ordering
  if factor == 'wf':
    if 'lf' in data[factor]: fLabels = ['hf', 'lf']
    elif 'LF' in data[factor]: fLabels = ['HF', 'LF']
  elif factor == 'conc':
    if 'lc' in data[factor]: fLabels = ['lc', 'hc']
    elif 'LC' in data[factor]: fLabels = ['LC', 'HC']
  elif factor == 'sa': fLabels = ['speed', 'accuracy']
  elif factor == 'strength':
    if 'weak' in data[factor]: fLabels = ['weak', 'strong']
    elif 'Weak' in data[factor]: fLabels = ['Weak', 'Strong']
  elif factor == 'length':
    if 'short' in data[factor]: fLabels = ['short', 'long']
    elif 'Short' in data[factor]: fLabels = ['Short', 'Long']
  elif factor == 'delay': fLabels = ['Immed', 'Delay']
  #elif factor == 'cond': fLabels = ['weak', 'strong', 'both', 'lure']
  else: fLabels = np.unique(data[factor])
  print 'fLabels: ' + str(fLabels)

  modelSubjects = np.arange(nSubj)
  for o, oldnew in enumerate(oldnews):
    for f, fact in enumerate(fLabels):
      subjCondition = np.unique(data[data[factor] == fact]['subj'])
      allSubjects = list(np.unique(data['subj']))
      subjIndices = [allSubjects.index(subject) for subject in subjCondition] #create the subject indices based on the condition
      subjectList = zip(subjIndices, subjCondition) #contains indices and subject numbers

      subjs = np.array([np.mean(model[sbj][:,np.where((data[data['subj'] == subject][factor] == fact) & (data[data['subj'] == subject]['type'] == oldnew))[0]]['corr'], axis = 1) for sbj, subject in subjectList])
      #reverse for lure items (get the FAR)
      if (oldnew == 'Lure') | (oldnew == 'lure'): subjs = 1.0 - subjs
      rate = np.nanmean(subjs, axis = 0)
      #edge corrections
      if zROC == True:
        total = np.array([len(data[(data['subj'] == subject) & (data[factor] == fact) & (data['type'] == oldnew)][corr_field]) for sbj, subject in subjectList])
        total = total.reshape((nSubj, 1))
        transform = ((subjs * total) + .5) / (total + 1) 
        zrate = np.mean(norm.ppf(transform), axis = 0) 
  
      nSubj = len(subjectList)
      RTs = np.zeros((nSubj,len(qs),nSamples))

      #loop over subjects and samples to get quantiles for each sample
      i = 0
      for sbj,subject in subjectList:
        for sample in xrange(nSamples):
          times = model[sbj][sample,(data[data['subj'] == subject]['type'] == oldnew) &
                                                    (data[data['subj'] == subject][factor] == fact) &
                                                    (model[sbj][sample]['RT'] > lower) &
                                                    (model[sbj][sample]['RT'] < upper) & 
                                                    (model[sbj][sample]['corr'] == 1)]['RT']
          if len(times) > 0: RTs[i,:,sample] = mstats.mquantiles(times, prob = qs)
          else: RTs[i,:,sample] = np.nan
        i += 1

      modelCorrRTs[o,f] = np.nanmean(RTs, axis = 0) #average over subjects, preserve quantiles and samples

      RTs = np.zeros((nSubj,len(qs),nSamples))
      i = 0
      for sbj,subject in subjectList:
        for sample in xrange(nSamples):
          times = model[sbj][sample,(data[data['subj'] == subject]['type'] == oldnew) &
                                                    (data[data['subj'] == subject][factor] == fact) &
                                                    (model[sbj][sample]['RT'] > lower) &
                                                    (model[sbj][sample]['RT'] < upper) & 
                                                    (model[sbj][sample]['corr'] == 0)]['RT']
          if len(times) > 0: RTs[i,:,sample] = mstats.mquantiles(times, prob = qs)
          else: RTs[i,:,sample] = np.nan
        i += 1
      modelErrorRTs[o,f] = np.nanmean(RTs, axis = 0)
      ratesModel[o,f] = rate
      if zROC == True: zRatesModel[o,f] = zrate

  if np.min(modelErrorRTs < lower): print 'Problem: lower'
  if np.max(modelErrorRTs > upper): print 'Problem: upper'

  if zROC == False: return ratesModel, modelCorrRTs, modelErrorRTs
  elif zROC == True: return ratesModel, zRatesModel, modelCorrRTs, modelErrorRTs

def synthSubjAverage_singleFactorConf(data, model, factor, qs = [.1, .5, .9], lower = .2, upper = 2.5, zROC = False):
  print 'Subject average single factor: ' + str(factor)
  nConf = len(np.unique(data['conf']))
  nSamples = model[0].shape[0]
  nSubj = len(np.unique(data['subj']))
  nLevels = len(np.unique(data[factor]))
  nOldnew = len(np.unique(data['type']))
  ratesModel = np.zeros((nConf,nOldnew,nLevels,nSamples))
  if zROC == True:
    zRatesModel = np.zeros((nConf,nOldnew,nLevels,nSamples))
    if 'correct' in data.dtype.fields: corr_field = 'correct'
    elif 'corr' in data.dtype.fields: corr_field = 'corr'
  modelCorrRTs = np.zeros((nConf,nOldnew,nLevels,len(qs),nSamples))
  modelErrorRTs = np.zeros((nConf,nOldnew,nLevels,len(qs),nSamples))

  if 'Target' in data['type']: oldnews = ['Target', 'Lure']
  elif 'target' in data['type']: oldnews = ['target', 'lure']
  elif 'left' in data['type']: oldnews = ['left', 'right']
  elif 'strong' in data['type']: oldnews = ['lure', 'weak', 'strong']
  else: oldnews = np.unique(data['type'])

  if 'response' in data.dtype.names: resp = 'response'
  elif 'resp' in data.dtype.names: resp = 'resp'
 
  #having predefined label orders ensures that there is no confusion about their ordering
  if factor == 'wf':
    if 'lf' in data[factor]: fLabels = ['hf', 'lf']
    elif 'LF' in data[factor]: fLabels = ['HF', 'LF']
  elif factor == 'conc':
    if 'lc' in data[factor]: fLabels = ['lc', 'hc']
    elif 'LC' in data[factor]: fLabels = ['LC', 'HC']
  elif factor == 'sa': fLabels = ['speed', 'accuracy']
  elif factor == 'strength':
    if 'weak' in data[factor]: fLabels = ['weak', 'strong']
    elif 'Weak' in data[factor]: fLabels = ['Weak', 'Strong']
  elif factor == 'length':
    if 'short' in data[factor]: fLabels = ['short', 'long']
    elif 'Short' in data[factor]: fLabels = ['Short', 'Long']
  elif factor == 'delay': fLabels = ['Immed', 'Delay']
  elif factor == 'cond': fLabels = ['weak', 'strong', 'both', 'lure']
  else: fLabels = np.unique(data[factor])
  print 'fLabels: ' + str(fLabels)

  for c in xrange(nConf):
    for o, oldnew in enumerate(oldnews):
      for f, fact in enumerate(fLabels):

        subjs = np.zeros((nSubj,nSamples))
        for sbj,subject in enumerate(np.unique(data['subj'])):
          for sample in xrange(nSamples):
            subjs[sbj,sample] = np.mean(model[sbj][sample,(data[data['subj'] == subject]['type'] == oldnew) &
                                                    (data[data['subj'] == subject][factor] == fact) &
                                                    (model[sbj][sample]['conf'] == c)]['corr'])
        #subjs = np.array([np.mean(model[sbj][:,np.where((data[data['subj'] == subject][factor] == fact) & (data[data['subj'] == subject]['type'] == oldnew) & (model[sbj]['conf'] == c))[0]]['corr'], axis = 1) for sbj, subject in enumerate(np.unique(data['subj']))])
        #reverse for lure items (get the FAR)
        if (oldnew == 'Lure') | (oldnew == 'lure'): subjs = 1.0 - subjs
        rate = np.mean(subjs, axis = 0)
        #edge corrections
        if zROC == True:
          total = np.array([len(data[(data['subj'] == subject) & (data[factor] == fact) & (data['type'] == oldnew)][corr_field]) for sbj, subject in enumerate(np.unique(data['subj']))])
          total = total.reshape((nSubj, 1))
          transform = ((subjs * total) + .5) / (total + 1)
          zrate = np.mean(norm.ppf(transform), axis = 0)
  
        RTs = np.zeros((nSubj,len(qs),nSamples))

        #loop over subjects and samples to get quantiles for each sample
        for sbj,subject in enumerate(np.unique(data['subj'])):
          for sample in xrange(nSamples):
            RTs[sbj,:,sample] = mstats.mquantiles(model[sbj][sample,(data[data['subj'] == subject]['type'] == oldnew) &
                                                    (data[data['subj'] == subject][factor] == fact) &
                                                    (model[sbj][sample]['conf'] == c) & 
                                                    (model[sbj][sample]['RT'] > lower) &
                                                    (model[sbj][sample]['RT'] < upper) & 
                                                    (model[sbj][sample]['corr'] == 1)]['RT'], prob = qs)

        modelCorrRTs[c,o,f] = np.mean(RTs, axis = 0) #average over subjects, preserve quantiles and samples

        RTs = np.zeros((nSubj,len(qs),nSamples))
        for sbj,subject in enumerate(np.unique(data['subj'])):
          for sample in xrange(nSamples):
            RTs[sbj,:,sample] = mstats.mquantiles(model[sbj][sample,(data[data['subj'] == subject]['type'] == oldnew) &
                                                    (data[data['subj'] == subject][factor] == fact) &
                                                    (model[sbj][sample]['conf'] == c) & 
                                                    (model[sbj][sample]['RT'] > lower) &
                                                    (model[sbj][sample]['RT'] < upper) & 
                                                    (model[sbj][sample]['corr'] == 0)]['RT'], prob = qs)
        modelErrorRTs[c,o,f] = np.mean(RTs, axis = 0)
        ratesModel[c,o,f] = rate
        if zROC == True: zRatesModel[c,o,f] = zrate

  if np.min(modelErrorRTs < lower): print 'Problem: lower'
  if np.max(modelErrorRTs > upper): print 'Problem: upper'

  if zROC == False: return ratesModel, modelCorrRTs, modelErrorRTs
  elif zROC == True: return ratesModel, zRatesModel, modelCorrRTs, modelErrorRTs 

def synthSubjAverage_doubleFactor(data, model, factor1, factor2, qs = [.1, .5, .9], lower = .2, upper = 2.5, zROC = False):
  print 'Subject average double factor: ' + str(factor1) + ', ' + str(factor2)
  nSamples = model[0].shape[0]
  nSubj = len(np.unique(data['subj']))
  nLevels1 = len(np.unique(data[factor1]))
  nLevels2 = len(np.unique(data[factor2]))
  nOldnew = len(np.unique(data['type']))
  ratesModel = np.zeros((nOldnew,nLevels1 * nLevels2,nSamples))
  if zROC == True:
    zRatesModel = np.zeros((nOldnew,nLevels1 * nLevels2,nSamples))
    if 'correct' in data.dtype.fields: corr_field = 'correct'
    elif 'corr' in data.dtype.fields: corr_field = 'corr'
  modelCorrRTs = np.zeros((nOldnew,nLevels1 * nLevels2,len(qs),nSamples))
  modelErrorRTs = np.zeros((nOldnew,nLevels1 * nLevels2,len(qs),nSamples))

  if 'Target' in data['type']: oldnews = ['Target', 'Lure']
  elif 'target' in data['type']: oldnews = ['target', 'lure']
  else: oldnews = np.unique(data['type'])

  if 'response' in data.dtype.names: resp = 'response'
  elif 'resp' in data.dtype.names: resp = 'resp'
 
  #having predefined label orders ensures that there is no confusion about their ordering
  if factor1 == 'wf':
    if 'lf' in data[factor1]: fLabels1 = ['hf', 'lf']
    elif 'LF' in data[factor1]: fLabels1 = ['HF', 'LF']
  elif factor1 == 'sa': fLabels1 = ['speed', 'accuracy']
  elif factor1 == 'strength': fLabels1 = ['weak', 'strong']
  elif factor1 == 'length':
    if 'short' in data[factor1]: fLabels1 = ['short', 'long']
    elif 'Short' in data[factor1]: fLabels1 = ['Short', 'Long']
  elif factor1 == 'delay': fLabels1 = ['Immed', 'Delay']
  else: fLabels1 = np.unique(data[factor1])

  if factor2 == 'wf':
    if 'lf' in data[factor2]: fLabels2 = ['hf', 'lf']
    elif 'LF' in data[factor2]: fLabels2 = ['HF', 'LF']
  elif factor2 == 'sa': fLabels2 = ['speed', 'accuracy']
  elif factor2 == 'strength': fLabels2 = ['weak', 'strong']
  elif factor2 == 'length':
    if 'short' in data[factor2]: fLabels2 = ['short', 'long']
    elif 'Short' in data[factor2]: fLabels2 = ['Short', 'Long']
  elif factor2 == 'delay': fLabels2 = ['Immed', 'Delay']
  else: fLabels2 = np.unique(data[factor2])

  print 'fLabels1: ' + str(fLabels1) + ' fLabels2: ' + str(fLabels2)

  for o, oldnew in enumerate(oldnews):
    f = 0 #f is used as the counter
    for f1, fact1 in enumerate(fLabels1):
      for f2, fact2 in enumerate(fLabels2):
        subjCondition = np.unique(data[(data[factor1] == fact1) & (data[factor2] == fact2)]['subj'])
        allSubjects = list(np.unique(data['subj']))
        subjIndices = [allSubjects.index(subject) for subject in subjCondition] #create the subject indices based on the condition
        subjectList = zip(subjIndices, subjCondition) #contains indices and subject numbers

        subjs = np.array([np.mean(model[sbj][:,np.where((data[data['subj'] == subject][factor1] == fact1) &
                                                        (data[data['subj'] == subject][factor2] == fact2) &
                                                        (data[data['subj'] == subject]['type'] == oldnew))[0]]['corr'], axis = 1) for sbj, subject in subjectList])
        #reverse for lure items (get the FAR)
        if (oldnew == 'Lure') | (oldnew == 'lure'): subjs = 1.0 - subjs
        rate = np.mean(subjs, axis = 0)
        #edge corrections
        if zROC == True:
          total = np.array([len(data[(data['subj'] == subject) & (data[factor1] == fact1) & (data[factor2] == fact2) & (data['type'] == oldnew)][corr_field]) for sbj, subject in subjectList])
          total = total.reshape((nSubj, 1))
          transform = ((subjs * total) + .5) / (total + 1) 
          zrate = np.mean(norm.ppf(transform), axis = 0) 
  
        RTs = np.zeros((nSubj,len(qs),nSamples))

        #loop over subjects and samples to get quantiles for each sample
        for sbj,subject in enumerate(np.unique(data['subj'])):
          for sample in xrange(nSamples):
            RTs[sbj,:,sample] = mstats.mquantiles(model[sbj][sample,(data[data['subj'] == subject]['type'] == oldnew) &
                                                    (data[data['subj'] == subject][factor1] == fact1) &
                                                    (data[data['subj'] == subject][factor2] == fact2) & 
                                                    (model[sbj][sample]['RT'] > lower) &
                                                    (model[sbj][sample]['RT'] < upper) & 
                                                    (model[sbj][sample]['corr'] == 1)]['RT'], prob = qs)

        modelCorrRTs[o,f] = np.mean(RTs, axis = 0) #average over subjects, preserve quantiles and samples

        RTs = np.zeros((nSubj,len(qs),nSamples))
        for sbj,subject in enumerate(np.unique(data['subj'])):
          for sample in xrange(nSamples):
            RTs[sbj,:,sample] = mstats.mquantiles(model[sbj][sample,(data[data['subj'] == subject]['type'] == oldnew) &
                                                    (data[data['subj'] == subject][factor1] == fact1) &
                                                    (data[data['subj'] == subject][factor2] == fact2) &
                                                    (model[sbj][sample]['RT'] > lower) &
                                                    (model[sbj][sample]['RT'] < upper) &
                                                    (model[sbj][sample]['corr'] == 0)]['RT'], prob = qs)
        modelErrorRTs[o,f] = np.mean(RTs, axis = 0)
        ratesModel[o,f] = rate
        if zROC == True: zRatesModel[o,f] = zrate
        f+=1

  if zROC == False: return ratesModel, modelCorrRTs, modelErrorRTs
  elif zROC == True: return ratesModel, zRatesModel, modelCorrRTs, modelErrorRTs 

def genRecovery(data, p, log_dens_like, theta, nJobs = -1, factors = -1):
  nSubj,nChains,nmc,nParams = theta.shape
  subjectList = np.unique(data['subj'])

  theta = theta.reshape((nSubj, nChains * nmc, nParams)) #collapse over chains/samples
  th = np.zeros((nSubj,nParams))
  for i in xrange(nSubj):
    th[i] = theta[i,np.random.randint(0,nChains*nmc)] #select a random parameter vector for each subject
  
  #generate fields for synthesized data
  fields = [('correct', np.uint8), ('RT', np.float32), ('subj', np.int16)]
  for factor in factors:
    fields.append((factor, data[factor].dtype))
  genData = np.zeros(len(data), fields)

  #generate predictions
  subjs = Parallel(n_jobs = nJobs)(delayed(log_dens_like)(th[subj], data[data['subj'] == subject], p, simulate = True) for subj, subject in enumerate(np.unique(data['subj'])))
  for s, subj in enumerate(subjs):
    subj_idx = data['subj'] == subjectList[s]
    genData['subj'][subj_idx] = subjectList[s]
    genData['RT'][subj_idx], genData['correct'][subj_idx] = subj[0].reshape(subj[0].size), subj[1].reshape(subj[1].size)

  #add factors
  for factor in factors:
    genData[factor] = data[factor]
  
  return genData, th

#this is used to create the likelihood structure (like) for a fit that does not have it
def genLike(theta, data, params, log_dens_like, directory, nJobs):
  nSubj, nChains, nmc, nParams = theta.shape
  like = np.zeros((nSubj,nChains,nmc))
  for i in xrange(nmc):
    like[:,:,i] = np.array(Parallel(n_jobs = nJobs)(delayed(log_dens_like)(theta[subj,:,i], data[data['subj'] == subject], params) for subj, subject in enumerate(np.unique(data['subj']))))
    if (i + 1) % 25 == 0: print '*',
    if (i + 1) % 500 == 0: print i
  return like
  
def update_progress(steps, total, status = ''):
  
    progress = np.around(steps / float(total), 4)
    barLength = 30 # Modify this to change the length of the progress bar
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "o"*block + "-"*(barLength-block), progress*100, str(steps) + ' / ' + str(total) + ' ', status)
    sys.stdout.write(text)
    sys.stdout.flush()
  



