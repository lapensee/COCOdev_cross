import os
import numpy as np
import cPickle as pickle
import de as de
import sys
sys.stdout.flush()
import pickle as cPickle
from scipy.stats import norm, binom
import time
from extract import *
from deFuncs import gelman_rubin
from functools import partial
from Multinomial import *
import plotSourceDev


def generateHyperPriors(params, dist, p):
  priors = {}
  for param in params:
    if 'gamma' in param:
      priors[param + '_mumu'] = .5
      priors[param + '_musigma'] = 2
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    if 'Uss' in param:
      priors[param + '_mumu'] = .9
      priors[param + '_musigma'] = .5
      if 'beta' in dist.keys():
        if p.index(param) in dist['beta']:
          priors[param + '_musigma'] = 2
          priors[param + '_sigmaalpha'] = .1
    priors[param + '_sigmabeta'] = 10
    if (param == 'rweak') | (param == 'rstrong') | (param == 'rWeak') | (param == 'rWeak2') | (param == 'rWeak3'):
      priors[param + '_mumu'] = 1.0
      priors[param + '_musigma'] = 1.0
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3.
    elif 'logr' in param:
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3.
    elif (param == 'prrweak') | (param == 'prrstrong'):
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = .1
      priors[param + '_sigmabeta'] = 10
    if param == 'rWeak':
      if p.index('rWeak') in dist['beta']:
        priors[param + '_mumu'] = .5
        priors[param + '_musigma'] = 2
        priors[param + '_sigmaalpha'] = 1.0
        priors[param + '_sigmabeta'] = 1.0 / 3
    if 'logUtt' in param:
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = .1
      priors[param + '_sigmabeta'] = 10
    if 'logVtt' in param:
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 25.
    elif 'Vtt' in param:
      priors[param + '_mumu'] = .2
      priors[param + '_musigma'] = .2
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    elif 'Vtt' in param:
      priors[param + '_mumu'] = .2
      priors[param + '_musigma'] = .2
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3.
    if param == 'Vaa':
      priors[param + '_mumu'] = .2
      priors[param + '_musigma'] = .2
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    if param == 'logVaa':
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3.
    if param == 'Vab':
      priors[param + '_mumu'] = .2
      priors[param + '_musigma'] = .2
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    if param == 'logVab':
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10.0
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    if param == 'logVac':
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10.0
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0
    if 'logVss' in param:
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    elif 'Vss' in param:
      priors[param + '_mumu'] = .2
      priors[param + '_musigma'] = .2
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    if 'VsuProp' in param:
      priors[param + '_mumu'] = .5
      priors[param + '_musigma'] = .5
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0
    elif 'logVsu' in param:
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    elif 'Vsu' in param:
      priors[param + '_mumu'] = .2
      priors[param + '_musigma'] = .2
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0
    if 'logbgNoise' in param:
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    elif 'bgNoise' in param:
      priors[param + '_mumu'] = .1
      priors[param + '_musigma'] = 1.0
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    if param == 'logVti':
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    elif param == 'Vti':
      priors[param + '_mumu'] = .01
      priors[param + '_musigma'] = .1
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    if param == 'Vsu':
      priors[param + '_mumu'] = .01
      priors[param + '_musigma'] = .1
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    elif 'crit' in param:
      priors[param + '_mumu'] = 0.0
      priors[param + '_musigma'] = 1.0
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3.0
    if param == 'rStrong':
      priors[param + '_mumu'] = 1.5
      priors[param + '_musigma'] = 1.0
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0
    #for the source memory paper: used stronger priors here for logrStrong due to the lack of constraint in the third experiment
    elif param == 'logrStrong':
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = .5
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 25.
    if param == 'mHF':
      priors[param + '_mumu'] = 500
      priors[param + '_musigma'] = 500
      priors[param + '_sigmaalpha'] = .1
      priors[param + '_sigmabeta'] = 10
    if param == 'Uaa':
      priors[param + '_mumu'] = .5
      priors[param + '_musigma'] = 2
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3.
    if param == 'nSource':
      priors[param + '_mumu'] = 1000
      priors[param + '_musigma'] = 1000
      priors[param + '_sigmaalpha'] = .1
      priors[param + '_sigmabeta'] = 10
    if 'logm' in param:
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = .1
      priors[param + '_sigmabeta'] = 10 / 3.
    if param == 'lognSource':
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = .1
      priors[param + '_sigmabeta'] = 10 / 3.
    if param == 'lognItem':
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = .1
      priors[param + '_sigmabeta'] = 10 / 3.
    if param in ['mSourceProp', 'nSourceProp']:
      priors[param + '_mumu'] = .25
      priors[param + '_musigma'] = .25
      priors[param + '_sigmaalpha'] = 1
      priors[param + '_sigmabeta'] = 1  / 5.
    elif 'logVcc' in param:
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 100
      priors[param + '_sigmaalpha'] = .1
      priors[param + '_sigmabeta'] = 10
    if param == 'Uss':
      priors[param + '_mumu'] = .5
      priors[param + '_musigma'] = .5
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3
    elif param in ['critR_I0', 'critR_I1', 'critR_I3', 'critR_I4', 'critR_S0', 'critR_S1', 'critR_S3', 'critR_S4', 'critR_G0', 'critR_G1', 'critR_G3', 'critR_G4']:
      priors[param + '_mumu'] = .5
      priors[param + '_musigma'] = .5
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3.
    if param[0] == 'd':
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 10
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3.
    if 'critR_G2' in param:    
      priors[param + '_mumu'] = 0
      priors[param + '_musigma'] = 1
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3.
    elif 'critR_G' in param:
      priors[param + '_mumu'] = .5
      priors[param + '_musigma'] = .5
      priors[param + '_sigmaalpha'] = 1.0
      priors[param + '_sigmabeta'] = 1.0 / 3.

  return priors

def startPoints(params):
  starts = {}
  for param in params:
    if 'r' in param: starts[param] = .8
    if 'Uss' in param: starts[param] = .95
    if 'Vtt' in param: starts[param] = .1
    elif 'logVtt' in param: starts[param] = (-5, 1.0)
    if 'Vss' in param: starts[param] = .1
    elif 'logVss' in param: starts[param] = np.log(.024)
    if 'logVti' in param: starts[param] = (-8.00, 1.5)
    if 'crit' in param: starts[param] = -.2
    if 'bgNoise' in param: starts[param] = .1
    if 'logbgNoise' in param: starts[param] = -1.0
    if 'VsuLF' in param: starts[param] = .01
    elif 'logVsuLF' in param: starts[param] = np.log(.01)
    if 'VsuHF' in param: starts[param] = .1
    elif 'logVsuHF' in param: starts[param] = np.log(.1)
    if 'logUtt' in param: starts[param] = np.log(1.1)
    if 'Vaa' in param: starts[param] = .00005
    #if param == 'logVaa': starts[param] = np.log(.001)
    #if param == 'logVaa': starts[param] = -10.0

    if 'Vab' in param: starts[param] = .0005
    #if param == 'logVab': starts[param] = np.log(.002)
    #if param == 'logVab': starts[param] = -15.0
    if param == 'Vsu': starts[param] = .001
    #elif param == 'logVsu': starts[param] = -4.0
    elif param == 'logVsu': starts[param] = (-10.0, 1.0)
    if param == 'mHF': starts[param] = 50
    if param == 'rStrong': starts[param] = 1.8
    if param == 'Uaa': starts[param] = .6
    if param == 'nSource': starts[param] = 2000.0
    if param == 'lognSource': starts[param] = 12
    if param == 'lognItem': starts[param] = 20
    #if param == 'lognItem': starts[param] = 6
    if param == 'nSourceProp': starts[param] = .22
    if param in ['logm', 'logmSource']: starts[param] = 4.0
    #if param == 'nSource': starts[param] = 20.0
    if param == 'critSource': starts[param] = 0.0
    if param == 'logVcc': starts[param] = -1.0
    if param == 'mSourceProp': starts[param] = .15

    if param == 'logVtt': starts[param] = (-2.5, 1.0)
    if param == 'rWeak': starts[param] = .8
    elif param == 'logrWeak': starts[param] = (-.1, 1.0)
    elif param == 'rWeak2': starts[param] = 1.0
    elif param == 'logrWeak2': starts[param] = (.25, 1.0)
    elif param == 'logrWeak3': starts[param] = (.1, 1.0)
    elif param == 'logrStrong': starts[param] = (-.29, .25)
    if param == 'logmHF': starts[param] = (7.5, 1.0)
    if param == 'logmHFSource': starts[param] = 6.4
    if param == 'crit': starts[param] = -.4
    elif param == 'crit2': starts[param] = (.3, .05)
    if param == 'logVaa': starts[param] = (-7.5, 1.5)
    if param == 'logVab': starts[param] = (-2.5, 1.5)
    if param == 'logVac': starts[param] = (-1.0, 1.0)
    if param == 'logbgNoise': starts[param] = -2.0

    if param == 'critR_I2': starts[param] = (-.25, .5)
    if param == 'critR_S2': starts[param] = (0, .5)

    if param in ['critR_I0', 'critR_S0', 'critR_G0']: starts[param] = (.07, .025)
    if param in ['critR_I1', 'critR_S1', 'critR_G1']: starts[param] = (.15, .025)

    if param in ['critR_I3', 'critR_S3', 'critR_G3']: starts[param] = (.15, .025)
    if param in ['critR_I4', 'critR_S4', 'critR_G4']: starts[param] = (.3, .025)
  
    if param == 'logVsu': starts[param] = -10

    if param[0] == 'd': starts[param] = (.2,1.0)

    #guessing criteria
    if 'critR_G2' in param: starts[param] = (.1, 1.0)
    elif 'critR_G0' in param: starts[param] = (.5, .05)
    elif 'critR_G1' in param: starts[param] = (.2, .05)
    elif 'critR_G3' in param: starts[param] = (.2, .05)
    elif 'critR_G4' in param: starts[param] = (.5, .05)
    
    if param == 'gamma': starts[param] = (.997, .03)

  return starts



def paramNamesSourceDevelopment(data, Uaa = False, informed = False, Vcc = False, rWeak = False, m = False, UssSource = True, Vti = True, Vsu = True, VaaVab = True, logr = False, prop = False):

  dist = {}
  dist['dtnorm'], dist['norm'] = [], []
  dist['dtnorm01'] = []
  if informed == True: dist['kde'] = []

  Uss = []
  Vtt = []
  Vss = []

  crit = ['crit', 'critSource']

  if Vsu == True: Vsu = ['logVsu']
  else: Vsu = []
  if Vti == False: Vti = []
  else: Vti = ['logVti']

  if VaaVab == True:
    Vaa = ['logVaa']
    Vab = ['logVab']
  else:
    Vaa, Vab = [], []

  if Vcc == True: Vcc = ['logVcc']
  else: Vcc = []

  r = ['logrWeak', 'logrStrong']

  m = ['logmHF', 'logmHFSource']
  prop = []

  if Uaa == False: Uaa = []
  else:
    Uaa = ['Uaa']
    Vaa = []
  Vac = ['logVac']

  
  params = crit + Vsu + Vti + Vtt + Vss + Vaa + Vab + Vac + r + m + Uaa + prop + Vcc + bgNoise + Uss
  dtnorm = []
  dtnorm01 = Uaa + Uss
  dtnorm_2 = prop 
  norm = crit + Vsu + Vti + Vtt + Vss + Vaa + Vab + Vac + m + Vcc + bgNoise + r

  #with informed priors you have to double list the KDE parameters. Subject level is evaluated analytically
  if informed == True: kde = Vsu + Vti + Vtt + Vss + Vaa + Vab + m + n + ['rStrong'] + Uaa + prop 

  for param in params:
    if param in dtnorm: dist['dtnorm'] += [params.index(param)]
    if param in norm: dist['norm'] += [params.index(param)]
    if 'dtnorm01' in dist.keys():
      if param in dtnorm01: dist['dtnorm01'] += [params.index(param)]
    if 'kde' in dist.keys():
      if param in kde: dist['kde'] += [params.index(param)]

  return params, dist

def paramNamesSourceDevelopmentFreq(data, **kwargs):

  dist = {}
  dist['dtnorm'], dist['norm'] = [], []
  dist['beta'] = []

  #item recognition parameters
  Uss = []
  Vtt = []
  Vss = []
  Vsu = ['logVsu']
  Vti = ['logVti']
  crit = ['crit', 'critSource']

  #source memory parameters
  Uaa = []
  if kwargs.get('Uaa', False) == True: Uaa = ['Uaa']

  Vaa, Vab = [], []
  VaaVab = kwargs.get('VaaVab', 'one')
  if VaaVab == True:
    Vaa = ['logVaa']
    Vab = ['logVab']
  elif VaaVab == 'one': #Vab = Vaa in this case
    Vaa = ['logVaa']

  Vcc, Vac = [], []
  if kwargs.get('Vcc', False) == True: Vcc = ['logVcc']
  if kwargs.get('Vac', True) == True: Vac = ['logVac']

  m = []
  if kwargs.get('prop', False) == True:
    #prop = ['nSourceProp', 'mSourceProp']
    prop = ['nSourceProp']
  else: prop = []

  #learning rate parameters
  rStrong = ['logrStrong']
  logr = kwargs.get('logr', False) 
  if logr == False: r = []
  elif logr == True: r = ['logrWeak']
  elif logr == 'beta': r = ['rWeak'] #if it's a beta, don't do a log transform

  #gamma = ['gamma']
  gamma = []  

  params = crit + Vsu + Vti + Vtt + Vss + Vaa + Vab + r + Uaa + prop + Vcc + Uss + Vac + gamma + rStrong
  dtnorm = []
  beta = Uaa + Uss + prop + gamma
  norm = crit + Vsu + Vti + Vtt + Vss + Vaa + Vab + m + Vcc + Vac + rStrong
  
  if logr == True: norm += r
  elif logr == 'beta': beta += r

  #with informed priors you have to double list the KDE parameters. Subject level is evaluated analytically
  if kwargs.get('informed', False) == True:
    dist['kde'] = []
    kde = Vsu + Vti + Vtt + Vss + Vaa + Vab + m + n + ['rStrong'] + Uaa + prop 

  if len(beta) == 0: dist.pop('beta', None)
  if len(dtnorm) == 0: dist.pop('dtnorm', None)

  for param in params:
    if param in dtnorm: dist['dtnorm'] += [params.index(param)]
    if param in norm: dist['norm'] += [params.index(param)]
    if 'beta' in dist.keys():
      if param in beta: dist['beta'] += [params.index(param)]
    if 'kde' in dist.keys():
      if param in kde: dist['kde'] += [params.index(param)]

  return params, dist


def logDensLikeSourceDevelopment(theta, data, p, pred = False, full = False, mixture = False, critMixture = True, fullyInformed = False):
  if len(theta.shape) == 1: theta = theta.reshape(1, len(theta))
  nChains = theta.shape[0]

  counts = np.zeros((5,3,2,3,1)) #type (lure, sourceAweak, sourceBweak, sourceAstrong, sourceBstrong) x cond x wf
  mus = np.zeros((9,3,2,nChains))
  sigmas = np.zeros((9,3,2,nChains))
  crit = np.zeros((9,3,2,nChains))

  crit[0:4] = theta[:,p.index('critSource')]
  crit[4:7] = theta[:,p.index('crit')]
  crit[7:9] = theta[:,p.index('critSource')] #source information for lures!

  if 'Vss' in p: Vss = theta[:,p.index('Vss')]
  elif 'logVss' in p: Vss = np.exp(theta[:,p.index('logVss')])
  else: Vss = .1
  if 'Vtt' in p: Vtt = theta[:,p.index('Vtt')]
  elif 'logVtt' in p: Vtt = np.exp(theta[:,p.index('logVtt')])
  else: Vtt = .02
  
  #fixed parameters
  r = np.ones(nChains)
  if 'rWeak' in p: r = theta[:,p.index('rWeak')]
  elif 'logrWeak' in p: r = np.exp(theta[:,p.index('logrWeak')])
  if 'rStrong' in p: rStrong = r * theta[:,p.index('rStrong')]
  else: rStrong = r * np.exp(theta[:,p.index('logrStrong')])

  Usu = 0
  Uti = 0
  Utt = 1.0
  Uss = 1.0

  Ucc = 1.0 #source match fixed to 1.0
  if 'Uaa' not in p: Uaa = 1.0
  else: Uaa = theta[:,p.index('Uaa')]
  Ubb = Uaa
  Uaa, Ubb = 1.0, 1.0
  Uab = 0.0
  Uba = 0.0
  if 'Vti' in p: Vti = theta[:,p.index('Vti')]
  elif 'logVti' in p: Vti = np.exp(theta[:,p.index('logVti')])
  else: Vti = 0
  if 'Vsu' in p: Vsu = theta[:,p.index('Vsu')]
  elif 'logVsu' in p: Vsu = np.exp(theta[:,p.index('logVsu')])
  else: Vsu = .0001

  if 'Vaa' in p: Vaa = theta[:,p.index('Vaa')]
  elif 'logVaa' in p: Vaa = np.exp(theta[:,p.index('logVaa')])
  else: Vaa = Vss
  if 'Vab' in p: Vab = theta[:,p.index('Vab')]
  elif 'logVab' in p: Vab = np.exp(theta[:,p.index('logVab')])
  else: Vab = Vsu
  Vbb = Vaa
  Vba = Vab
  if 'logVcc' in p: Vcc = np.exp(theta[:,p.index('logVcc')])
  else: Vcc = 0
  if 'logVac' in p: Vac = np.exp(theta[:,p.index('logVac')])
  else: Vac = 0
  Vab = Vac

  Uac = 0
  Vac = 0
  Ubc = 0
  Vbc = 0

  if 'logbgNoise' in p: bgNoiseSource = np.exp(theta[:,p.index('logbgNoise')])
  else: bgNoiseSource = 0.0

  for w, wf in enumerate(['HF', 'LF']):
    wf_idx = data['wf'] == wf

    if 'logm' in p: m = np.exp(theta[:,p.index('logm')])
    else: m = 0
    if 'logmSource' in p: mSource = np.exp(theta[:,p.index('logmSource')])
    else: mSource = 0
    if wf == 'HF':
      if 'mHF' in p: m += theta[:,p.index('mHF')]
      elif 'logmHF' in p: m += np.exp(theta[:,p.index('logmHF')])
      if 'logmHFSource' in p: mSource += np.exp(theta[:,p.index('logmHFSource')])
      if 'mSourceProp' in p: mSource += m * theta[:,p.index('mSourceProp')]

    for c, cond in enumerate(['pure', 'mixed', 'long']):
  
      l = 8
      if cond == 'pure': r2 = r
      elif cond == 'mixed': r2 = rStrong
      else: l = 16
      rAvg = np.mean(np.vstack((r,r2)), axis = 0)

      cond_idx = data['cond'] == cond
      
      #item recognition
      UoldWeak = r * Uss * Utt * Ucc
      UoldStrong = r2 * Uss * Utt * Ucc
      UoldAvg = rAvg * Uss * Utt * Ucc

      selfMatch = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vcc + Ucc**2)) - (Utt**2 * Uss**2 * Ucc**2)
      matchOthers = ((Vti + Uti**2) * (Vss + Uss**2) * (Vcc + Ucc**2)) - (Uti**2 * Uss**2 * Ucc**2)
      contextNoise = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vcc + Ucc**2)) - (Utt**2 * Usu**2 * Ucc**2)
      bgNoise = .05

      SoldWeak = np.sqrt(((r ** 2) * selfMatch) + (((l/2 - 1) * (r ** 2)) * matchOthers) + ((l/2 * (r2 ** 2)) * matchOthers) + (m * contextNoise) + bgNoise)
      SoldStrong = np.sqrt(((r2 ** 2) * selfMatch) + ((l/2 * (r ** 2)) * matchOthers) + (((l/2 - 1) * (r2 ** 2)) * matchOthers) + (m * contextNoise) + bgNoise)
      Snew = np.sqrt(((l/2 * (r ** 2)) * matchOthers) + (l/2 * (r2 ** 2) * matchOthers) + (m * contextNoise) + bgNoise)
      SoldAvg = np.mean(np.vstack((SoldWeak, SoldStrong)), axis = 0)

      dWeak = UoldWeak/Snew
      SWeak = SoldWeak/Snew
      dStrong = UoldStrong/Snew
      SStrong = SoldStrong/Snew
      d = UoldAvg/Snew
      S = SoldAvg/Snew

      alpha = (S**2 + 1) / (2 * S**2)
      beta = (S**2 + 3) / (4 * S**2)

      muOldWeak = (d*dWeak * alpha) - (((d**2 / 2) * beta) + np.log(S))
      muOldStrong = (d*dStrong * alpha) - (((d**2 / 2) * beta) + np.log(S))
      muNew = -(((d**2 / 2) * beta) + np.log(S))
    
      sigmaOldWeak = d * alpha * SWeak
      sigmaOldStrong = d * alpha * SStrong
      sigmaNew = d * alpha

      mus[4,c,w], mus[5,c,w], mus[6,c,w] = muNew, muOldWeak, muOldStrong
      sigmas[4,c,w], sigmas[5,c,w], sigmas[6,c,w] = sigmaNew, sigmaOldWeak, sigmaOldStrong

      #counts takes the FAR here since that's the only item related decision!
      for i in xrange(3): counts[4,c,w,i] = np.sum(data[wf_idx & cond_idx & (data['type'] == 'lure')]['response'] == i)

      #source memory
      UA = r * Utt * Uss * Uaa
      UB = r * Utt * Uss * Ubb
      #UA and UB for lures
      UA0 = r * Uti * Uss * Uaa 
      UB0 = r * Uti * Uss * Ubb

      UAStrong = r2 * Utt * Uss * Uaa
      UBStrong = r2 * Utt * Uss * Ubb

      UAAvg = rAvg * Utt * Uss * Uaa
      UBAvg = rAvg * Utt * Uss * Ubb

      selfMatchAA = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vaa + Uaa**2)) - (Utt**2 * Uss**2 * Uaa**2)
      selfMatchAB = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vab + Uab**2)) - (Utt**2 * Uss**2 * Uab**2)
      selfMatchBA = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vba + Uba**2)) - (Utt**2 * Uss**2 * Uba**2)
      selfMatchBB = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vbb + Ubb**2)) - (Utt**2 * Uss**2 * Ubb**2)
      matchOthersAA = ((Vti + Uti**2) * (Vss + Uss**2) * (Vaa + Uaa**2)) - (Uti**2 * Uss**2 * Uaa**2)
      matchOthersAB = ((Vti + Uti**2) * (Vss + Uss**2) * (Vab + Uab**2)) - (Uti**2 * Uss**2 * Uab**2)
      matchOthersBB = ((Vti + Uti**2) * (Vss + Uss**2) * (Vbb + Ubb**2)) - (Uti**2 * Uss**2 * Ubb**2)
      matchOthersBA = ((Vti + Uti**2) * (Vss + Uss**2) * (Vba + Uba**2)) - (Uti**2 * Uss**2 * Uba**2)

      #what I discovered: you can't do it without context noise or background noise! The model can't even capture the strength effect!!!
      contextNoiseAA = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vaa + Uaa**2)) - (Utt**2 * Usu**2 * Uaa**2)
      contextNoiseAB = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vab + Uab**2)) - (Utt**2 * Usu**2 * Uab**2)
      contextNoiseBB = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vbb + Ubb**2)) - (Utt**2 * Usu**2 * Ubb**2)
      contextNoiseBA = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vba + Uba**2)) - (Utt**2 * Usu**2 * Uba**2)
      contextNoiseAC = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vac + Uac**2)) - (Utt**2 * Usu**2 * Uac**2) #item level context noise
      contextNoiseBC = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vac + Ubc**2)) - (Utt**2 * Usu**2 * Ubc**2) #item level context noise
      bgNoiseAA = .05 + bgNoiseSource
      bgNoiseAB = .05 + bgNoiseSource
      bgNoiseBB = .05 + bgNoiseSource
      bgNoiseBA = .05 + bgNoiseSource

      V_AA = ((r ** 2) * selfMatchAA) + (((l/4 - 1) * (r ** 2)) * matchOthersAA) + (((l/4) * (r**2)) * matchOthersAB)
      V_AA += ((l/4) * (r2**2) * matchOthersAA) + ((l/4) * (r2**2) * matchOthersAB)
      V_AA += bgNoiseAA + bgNoiseAB + (mSource * contextNoiseAA) + (mSource * contextNoiseAB) + (m * contextNoiseAC)

      V_BA = ((r ** 2) * selfMatchBA) + (((l/4 - 1) * (r ** 2)) * matchOthersBA) + (((l/4) * (r**2)) * matchOthersBB)
      V_BA += ((l/4) * (r2**2) * matchOthersBA) + ((l/4) * (r2**2) * matchOthersBB)
      V_BA += bgNoiseBA + bgNoiseBB + (mSource * contextNoiseBA) + (mSource * contextNoiseBB) + (m * contextNoiseBC)

      V_AB = ((r ** 2) * selfMatchAB) + (((l/4 - 1) * (r ** 2)) * matchOthersAB) + (((l/4) * (r**2)) * matchOthersAA)
      V_AB += ((l/4) * (r2**2) * matchOthersAB) + ((l/4) * (r2**2) * matchOthersAA)
      V_AB += bgNoiseAB + bgNoiseAA + (mSource * contextNoiseAB) + (mSource * contextNoiseAA) + (m * contextNoiseAC)

      V_BB = ((r ** 2) * selfMatchBB) + (((l/4 - 1) * (r ** 2)) * matchOthersBB) + (((l/4) * (r**2)) * matchOthersBA)
      V_BB += ((l/4) * (r2**2) * matchOthersBB) + ((l/4) * (r2**2) * matchOthersBA)
      V_BB += bgNoiseBB + bgNoiseBA + (mSource * contextNoiseBB) + (mSource * contextNoiseBA) + (m * contextNoiseBC)

      V_A0 = ((l/4 * (r ** 2)) * matchOthersAA) + (((l/4) * (r**2)) * matchOthersAB)
      V_A0 += ((l/4) * (r2**2) * matchOthersAA) + ((l/4) * (r2**2) * matchOthersAB)
      V_A0 += bgNoiseAA + bgNoiseAB + (mSource * contextNoiseAA) + (mSource * contextNoiseAB) + (m * contextNoiseAC)

      V_B0 = ((l/4 * (r ** 2)) * matchOthersBB) + (((l/4) * (r**2)) * matchOthersBA)
      V_B0 += ((l/4) * (r2**2) * matchOthersBB) + ((l/4) * (r2**2) * matchOthersBA)
      V_B0 += bgNoiseBB + bgNoiseBA + (mSource * contextNoiseBB) + (mSource * contextNoiseBA) + (m * contextNoiseBC)

      V_AAstrong = ((r2 ** 2) * selfMatchAA) + (((l/4 - 1) * (r2 ** 2)) * matchOthersAA) + (((l/4) * (r2**2)) * matchOthersAB)
      V_AAstrong += ((l/4) * (r**2) * matchOthersAA) + ((l/4) * (r**2) * matchOthersAB)
      V_AAstrong += bgNoiseAA + bgNoiseAB + (mSource * contextNoiseAA) + (mSource * contextNoiseAA)

      V_BAstrong = ((r2 ** 2) * selfMatchBA) + (((l/4 - 1) * (r2 ** 2)) * matchOthersBA) + (((l/4) * (r2**2)) * matchOthersBB)
      V_BAstrong += ((l/4) * (r**2) * matchOthersBA) + ((l/4) * (r**2) * matchOthersBB)
      V_BAstrong += bgNoiseBA + bgNoiseBB + (mSource * contextNoiseBA) + (mSource * contextNoiseBB)

      V_ABstrong = ((r2 ** 2) * selfMatchAB) + (((l/4 - 1) * (r2 ** 2)) * matchOthersAB) + (((l/4) * (r2**2)) * matchOthersAA)
      V_ABstrong += ((l/4) * (r**2) * matchOthersAB) + ((l/4) * (r**2) * matchOthersAA)
      V_ABstrong += bgNoiseAB + bgNoiseAA + (mSource * contextNoiseAB) + (mSource * contextNoiseAA)

      V_BBstrong = ((r2 ** 2) * selfMatchBB) + (((l/4 - 1) * (r2 ** 2)) * matchOthersBB) + (((l/4) * (r2**2)) * matchOthersBA)
      V_BBstrong += ((l/4) * (r**2) * matchOthersBB) + ((l/4) * (r**2) * matchOthersBA)
      V_BBstrong += bgNoiseBB + bgNoiseBA + (mSource * contextNoiseBB) + (mSource * contextNoiseBA)

      #difference between the sources

      S_A = np.sqrt(V_AA + V_AB)
      S_B = np.sqrt(V_BB + V_AB)
      S_AStrong = np.sqrt(V_AAstrong + V_BAstrong)
      S_BStrong = np.sqrt(V_BBstrong + V_ABstrong)
      S_A0 = np.sqrt(V_A0 + V_B0)
      S_B0 = np.sqrt(V_A0 + V_B0)
      SAvgWeak = np.mean(np.vstack((S_A, S_B)), axis = 0)
      SAvgStrong = np.mean(np.vstack((S_AStrong, S_BStrong)), axis = 0)
      SAvg = np.mean(np.vstack((S_A, S_B, S_AStrong, S_BStrong)), axis = 0)
         
      d = (UAAvg + UBAvg)/SAvg
      dWeak = (UA + UB)/SAvgWeak
      dStrong = (UAStrong + UBStrong)/SAvgStrong
      d0 = 0

      #just going to use the equal variance version here for now - can use an unequal variance version with unequal source strength     
      if fullyInformed == False:

        muA = dWeak * d/2
        muAStrong = dStrong * d/2
        muB = -(dWeak * d/2)
        muBStrong = -(dStrong * d/2)
        mu0A = d0 * d/2
        mu0B = -(d0 * d/2)

        #variance is solely due to expected strength! so for an uninformed model the expected variance is the same for both weak and strong items
        sigma = d
        sigmaStrong = d
        sigmaLure = d
      #below is the fully informed model = the case here is that subjects could use item strength to estimate the source strength
      else:
        muA = dWeak**2/2
        muAStrong = dStrong**2/2
        muB = -(dWeak**2)/2
        muBStrong = -(dStrong**2)/2

        d0 = dWeak #just assume that they use the d for weak items
        mu0A = d0 * d/2
        mu0B = -(d0 * d/2)

        sigma = dWeak
        sigmaStrong = dStrong
        sigmaLure = d0

      for i in xrange(3):
        counts[0,c,w,i] = np.sum(data[wf_idx & cond_idx & (data['type'] == 'weak') & (data['source'] == 0)]['response'] == i)
        counts[1,c,w,i] = np.sum(data[wf_idx & cond_idx & (data['type'] == 'weak') & (data['source'] == 1)]['response'] == i)
        if cond == 'mixed':
          counts[2,c,w,i] = np.sum(data[wf_idx & cond_idx & (data['type'] == 'strong') & (data['source'] == 0)]['response'] == i)
          counts[3,c,w,i] = np.sum(data[wf_idx & cond_idx & (data['type'] == 'strong') & (data['source'] == 1)]['response'] == i)

      mus[0,c,w], mus[1,c,w], mus[2,c,w], mus[3,c,w] = muA, muB, muAStrong, muBStrong
      sigmas[0,c,w], sigmas[1,c,w], sigmas[2,c,w], sigmas[3,c,w] = sigma, sigma, sigmaStrong, sigmaStrong

      mus[7,c,w], mus[8,c,w], sigmas[7,c,w], sigmas[8,c,w] = mu0A, mu0B, sigmaLure, sigmaLure

  #add predictions here later
  #likelihood calculation
  if pred == True:
    modelRates = 1.0 - norm.cdf(crit, loc = mus, scale = sigmas)
    modelRates = modelRates[0:7] #source 0-3, lure 4, weak strong HR 6-7
    modelRates[1], modelRates[3] = 1.0 - modelRates[1], 1.0 - modelRates[3]
    
    counts = counts.reshape(5,3,2,3) #get rid of the last dimension (length 1)
    dataRates = np.zeros((7,3,2))

    sm = np.sum(counts[0:1,:,:,0:2], axis = 3).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[0] = counts[0,:,:,0] / sm

    sm[sm == 0] = 1
    dataRates[1] = counts[1,:,:,1] / sm

    sm = np.sum(counts[2:3,:,:,0:2], axis = 3).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[2] = counts[2,:,:,0] / sm

    sm = np.sum(counts[3:4,:,:,0:2], axis = 3).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[3] = counts[3,:,:,1] / sm

    sm = np.sum(counts[4:5], axis = 3).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[4] = (counts[4,:,:,0] + counts[4,:,:,1]) / sm #FAR = (source A + source B) / CR

    #weird indexing below... if you use [0,1] for the first dimension, subsequent indices are off...
    sm = np.sum(counts[0:2], axis = (0,3)).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[5] = np.sum(counts[0:2,:,:,[0,1]], axis = (0,3)) / sm

    sm = np.sum(counts[2:4], axis = (0,3)).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[6] = np.sum(counts[2:4,:,:,[0,1]], axis = (0,3)) / sm
    
    return modelRates, dataRates
    
  elif pred == False:

    cdfs = norm.cdf(crit, loc = mus, scale = sigmas)
    #convert to prs
    prs = np.zeros((5,3,2,3,nChains))

    FAR, HR, strongHR = 1 - cdfs[4], 1 - cdfs[5], 1 - cdfs[6]
    CR = 1 - FAR

    #sourceA/sourceB judgments are weighted by the HR (source judgment is given only if it's recognized)
    prs[0,:,:,0] = HR * (1 - cdfs[0])
    prs[0,:,:,1] = HR * (cdfs[0])
    prs[0,:,:,2] = 1 - HR
    prs[1,:,:,0] = HR * (1 - cdfs[1])
    prs[1,:,:,1] = HR * (cdfs[1])
    prs[1,:,:,2] = 1 - HR

    prs[2,:,:,0] = strongHR * (1 - cdfs[2])
    prs[2,:,:,1] = strongHR * (cdfs[2])
    prs[2,:,:,2] = 1 - strongHR
    prs[3,:,:,0] = strongHR * (1 - cdfs[3])
    prs[3,:,:,1] = strongHR * (cdfs[3])
    prs[3,:,:,2] = 1 - strongHR

    prs[4,:,:,0] = FAR * (1 - cdfs[7])
    prs[4,:,:,1] = FAR * (cdfs[8])
    prs[4,:,:,2] = CR

    #make sure that no values are zero
    prs = ((prs * 1000000) + .5) / 1000001

    d = np.ones((5,3,2,nChains))
    for i in xrange(5):
      for j in xrange(2):
        for k in xrange(2):
          m = Multinomial(prs[i,j,k])
          d[i,j,k] = m.pmf(counts[i,j,k,:,0].reshape(3,1))
               
         
    if full == False:
      d = np.sum(np.log(d), axis = (0,1,2))
    else:
      d = d.reshape(5*3*2,nChains)
      d = np.swapaxes(d,0,1) #swap chains and conditions
    return d

def logDensLikeSourceDevelopmentFreq(theta, data, p, pred = False, full = False, mixture = False, critMixture = True, fullyInformed = False, simulate = False):
  if len(theta.shape) == 1: theta = theta.reshape(1, len(theta))
  nChains = theta.shape[0]

  counts = np.zeros((3,len(data)))
  mus = np.zeros((len(data),nChains))
  sigmas = np.zeros((len(data),nChains))
  musSource = np.zeros((len(data),nChains))
  sigmasSource = np.zeros((len(data),nChains))

  if 'Vss' in p: Vss = theta[:,p.index('Vss')]
  elif 'logVss' in p: Vss = np.exp(theta[:,p.index('logVss')])
  else: Vss = 0.0 #changed this to zero due to context drift
  if 'Vtt' in p: Vtt = theta[:,p.index('Vtt')]
  elif 'logVtt' in p: Vtt = np.exp(theta[:,p.index('logVtt')])
  else: Vtt = .02
 
  #n calculation
  age = data['age'][0] 
  n = age * 10950000
  nmillion = n / 1000000

  #fixed parameters
  r = np.ones(nChains)
  if 'rWeak' in p: r = theta[:,p.index('rWeak')].reshape(1,nChains)
  elif 'logrWeak' in p: r = np.exp(theta[:,p.index('logrWeak')]).reshape(1,nChains)
  if 'rStrong' in p: rStrong = r * (1 + theta[:,p.index('rStrong')].reshape(1,nChains)) #rStrong is a strength scale factor
  else: rStrong = r * (1 + np.exp(theta[:,p.index('logrStrong')]))

  Usu = 0
  Uti = 0
  Utt = 1.0
  UssBase = 1.0
  
  if 'gamma' in p: gamma = theta[:,p.index('gamma')].reshape(1,nChains)
  else: gamma = 1.0

  Ucc = 1.0 #source match fixed to 1.0
  if 'Uaa' not in p: Uaa = 1.0
  else: Uaa = theta[:,p.index('Uaa')].reshape(1,nChains)
  Ubb = Uaa
  Uaa, Ubb = 1.0, 1.0
  Uab = 0.0
  Uba = 0.0
  if 'Vti' in p: Vti = theta[:,p.index('Vti')].reshape(1,nChains)
  elif 'logVti' in p: Vti = np.exp(theta[:,p.index('logVti')]).reshape(1,nChains)
  else: Vti = 0
  if 'Vsu' in p: Vsu = theta[:,p.index('Vsu')].reshape(1,nChains)
  elif 'logVsu' in p: Vsu = np.exp(theta[:,p.index('logVsu')]).reshape(1,nChains)
  else: Vsu = .0001

  if 'Vaa' in p: Vaa = theta[:,p.index('Vaa')].reshape(1,nChains)
  elif 'logVaa' in p: Vaa = np.exp(theta[:,p.index('logVaa')]).reshape(1,nChains)
  else: Vaa = 0
  if 'Vab' in p: Vab = theta[:,p.index('Vab')].reshape(1,nChains)
  elif 'logVab' in p: Vab = np.exp(theta[:,p.index('logVab')]).reshape(1,nChains)
  else: Vab = Vaa
  Vbb = Vaa
  Vba = Vab
  if 'logVcc' in p: Vcc = np.exp(theta[:,p.index('logVcc')]).reshape(1,nChains)
  else: Vcc = 0
  if 'logVac' in p: Vac = np.exp(theta[:,p.index('logVac')]).reshape(1,nChains)
  else: Vac = Vaa
  Vbc = Vac

  Uac = 0
  Ubc = 0

  if 'nSourceProp' in p: nSourceProp = theta[:,p.index('nSourceProp')].reshape(1,nChains)
  else: nSourceProp = .10
  if 'mSourceProp' in p: mSourceProp = theta[:,p.index('mSourceProp')].reshape(1,nChains)
  else: mSourceProp = nSourceProp + 0.0

  nSource = np.log(n * nSourceProp)
  n2 = np.log(n - nSource)
  n = np.log(n)

  weak = data['type'] == 'weak'
  strong = data['type'] == 'strong'
  lure = data['type'] == 'lure'

  nTrials = len(np.unique(data['trial']))  
  trials = np.arange(nTrials).reshape(nTrials,1)
  
  Uss = UssBase * (gamma**trials) #context drift
  UssTest = [UssBase * (gamma**trials[0:t]) for t in xrange(nTrials)] #each entry is LL x chains

  #item recognition

  selfMatch = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vcc + Ucc**2)) - (Utt**2 * Uss**2 * Ucc**2)
  matchOthers = ((Vti + Uti**2) * (Vss + Uss**2) * (Vcc + Ucc**2)) - (Uti**2 * Uss**2 * Ucc**2)
  contextNoise = ((Vtt + Utt**2) * (Vsu + Usu**2) * (Vcc + Ucc**2)) - (Utt**2 * Usu**2 * Ucc**2)
  bgNoise = ((Vti + Uti**2) * (Vsu + Usu**2) * (Vcc + Ucc**2)) - (Uti**2 * Usu**2 * Ucc**2)
  
  matchOthersTest = np.array([np.sum(((UssTest[t] ** 2) * Vti) + ((Uti ** 2) * Vss) + (Vti * Vss), axis = 0) for t in xrange(nTrials)])

  #source memory

  selfMatchAA = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vaa + Uaa**2)) - (Utt**2 * Uss**2 * Uaa**2)
  selfMatchAB = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vab + Uab**2)) - (Utt**2 * Uss**2 * Uab**2)
  selfMatchBA = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vba + Uba**2)) - (Utt**2 * Uss**2 * Uba**2)
  selfMatchBB = ((Vtt + Utt**2) * (Vss + Uss**2) * (Vbb + Ubb**2)) - (Utt**2 * Uss**2 * Ubb**2)
  matchOthersAA = ((Vti + Uti**2) * (Vss + Uss**2) * (Vaa + Uaa**2)) - (Uti**2 * Uss**2 * Uaa**2)
  matchOthersAB = ((Vti + Uti**2) * (Vss + Uss**2) * (Vab + Uab**2)) - (Uti**2 * Uss**2 * Uab**2)
  matchOthersBB = ((Vti + Uti**2) * (Vss + Uss**2) * (Vbb + Ubb**2)) - (Uti**2 * Uss**2 * Ubb**2)
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
     
  
  for w, wf in enumerate(['HF', 'LF']):
    wf_idx = data['wf'] == wf

    for c, cond in enumerate(['pure', 'long', 'mixed']):
      cond_idx = data['cond'] == cond

      l = 8
      if cond in ['pure', 'long']: r2 = r
      elif cond == 'mixed': r2 = rStrong
      else: l = 16
      rAvg = np.mean(np.vstack((r,r2)), axis = 0)
      
      UoldWeak = r * Uss * Utt * Ucc
      UoldStrong = r2 * Uss * Utt * Ucc
      UoldAvg = rAvg * Uss * Utt * Ucc

      UA = r * Utt * Uss * Uaa
      UB = r * Utt * Uss * Ubb
      #UA and UB for lures
      UA0 = r * Uti * Uss * Uaa 
      UB0 = r * Uti * Uss * Ubb

      UAStrong = r2 * Utt * Uss * Uaa
      UBStrong = r2 * Utt * Uss * Ubb

      UAAvg = rAvg * Utt * Uss * Uaa
      UBAvg = rAvg * Utt * Uss * Ubb 

      for i in xrange(3):
        counts[i,wf_idx & cond_idx & lure] = (data[wf_idx & cond_idx & lure]['response'] == i) * 1
        counts[i,wf_idx & cond_idx & weak & (data['source'] == 0)] = (data[wf_idx & cond_idx & weak & (data['source'] == 0)]['response'] == i) * 1
        counts[i,wf_idx & cond_idx & weak & (data['source'] == 1)] = (data[wf_idx & cond_idx & weak & (data['source'] == 1)]['response'] == i) * 1
        if cond == 'mixed':
          counts[i,wf_idx & cond_idx & strong & (data['source'] == 0)] = (data[wf_idx & cond_idx & strong & (data['source'] == 0)]['response'] == i) * 1
          counts[i,wf_idx & cond_idx & strong & (data['source'] == 1)] = (data[wf_idx & cond_idx & strong & (data['source'] == 1)]['response'] == i) * 1

      if cond == 'mixed': types = ['weakA', 'weakB', 'strongA', 'strongB', 'lure']
      else: types = ['weakA', 'weakB', 'lure']
      for oldnew in types:
        if oldnew == 'weakA': type_idx = (data['type'] == 'weak') & (data['source'] == 0)
        elif oldnew == 'weakB': type_idx = (data['type'] == 'weak') & (data['source'] == 1)
        elif oldnew == 'strongA': type_idx = (data['type'] == 'strong') & (data['source'] == 0)
        elif oldnew == 'strongB': type_idx = (data['type'] == 'strong') & (data['source'] == 1)
        elif oldnew == 'lure': type_idx = (data['type'] == 'lure')

        #t: trials index - context match changes over trials
        t = data[cond_idx & wf_idx & type_idx]['trial']

        m = data[wf_idx & cond_idx & type_idx]['ageFreq'] * nmillion #log counts per million times nmillion times words have been experienced
        if len(m) == 0: print oldnew
        m = m.reshape(len(m),1)
        mSource = np.log(m * mSourceProp)
        m2 = np.log(m - mSource)
        m = np.log(m)
        
        SoldWeak = np.sqrt(((r ** 2) * selfMatch[t]) + (((l/2 - 1) * (r ** 2)) * matchOthers[t]) + ((l/2 * (r2 ** 2)) * matchOthers[t]) + (m * contextNoise) + (n * bgNoise))
        SoldStrong = np.sqrt(((r2 ** 2) * selfMatch[t]) + ((l/2 * (r ** 2)) * matchOthers[t]) + (((l/2 - 1) * (r2 ** 2)) * matchOthers[t]) + (m * contextNoise) + (n * bgNoise))
        Snew = np.sqrt(((l/2 * (r ** 2)) * matchOthers[t]) + (l/2 * (r2 ** 2) * matchOthers[t]) + (m * contextNoise) + (n * bgNoise))
        SoldAvg = np.mean(np.vstack((SoldWeak, SoldStrong)), axis = 0)

        Snew2 = ((l/2 * (r ** 2)) * matchOthers[t]) + (l/2 * (r2 ** 2) * matchOthers[t]) + (m * contextNoise) + (n * bgNoise)

        #have to reference the d parameters with t
        dWeak = UoldWeak[t]/Snew
        SWeak = SoldWeak/Snew
        dStrong = UoldStrong[t]/Snew
        SStrong = SoldStrong/Snew
        d = UoldAvg[t]/Snew
        S = SoldAvg/Snew

        alpha = (S**2 + 1) / (2 * S**2)
        beta = (S**2 + 3) / (4 * S**2)

        muOldWeak = (d*dWeak * alpha) - (((d**2 / 2) * beta) + np.log(S))
        muOldStrong = (d*dStrong * alpha) - (((d**2 / 2) * beta) + np.log(S))
        muNew = -(((d**2 / 2) * beta) + np.log(S))
      
        sigmaOldWeak = d * alpha * SWeak
        sigmaOldStrong = d * alpha * SStrong
        sigmaNew = d * alpha

        if 'weak' in oldnew: mu, sigma = muOldWeak, sigmaOldWeak
        elif 'strong' in oldnew: mu, sigma = muOldStrong, sigmaOldStrong
        elif oldnew == 'lure': mu, sigma = muNew, sigmaNew
        
        mus[wf_idx & cond_idx & type_idx], sigmas[wf_idx & cond_idx & type_idx] = mu, sigma

        V_AA = (nSource * bgNoiseAA) + (nSource * bgNoiseAB) + (n2 * bgNoiseAC) + (mSource * contextNoiseAA) + (mSource * contextNoiseAB) + (m2 * contextNoiseAC)
        V_AA += ((r ** 2) * selfMatchAA[t]) + (((l/4 - 1) * (r ** 2)) * matchOthersAA[t]) + (((l/4) * (r**2)) * matchOthersAB[t])
        V_AA += ((l/4) * (r2**2) * matchOthersAA[t]) + ((l/4) * (r2**2) * matchOthersAB[t])

        V_BA = (nSource * bgNoiseBA) + (nSource * bgNoiseBB) + (n2 * bgNoiseBC) + (mSource * contextNoiseBA) + (mSource * contextNoiseBB) + (m2 * contextNoiseBC)
        V_BA += ((r ** 2) * selfMatchBA[t]) + (((l/4 - 1) * (r ** 2)) * matchOthersBA[t]) + (((l/4) * (r**2)) * matchOthersBB[t])
        V_BA += ((l/4) * (r2**2) * matchOthersBA[t]) + ((l/4) * (r2**2) * matchOthersBB[t])

        V_AB = (nSource * bgNoiseAB) + (nSource * bgNoiseAB) + (n2 * bgNoiseAC) + (mSource * contextNoiseAA) + (mSource * contextNoiseAB) + (m2 * contextNoiseAC)
        V_AB += ((r ** 2) * selfMatchAB[t]) + (((l/4 - 1) * (r ** 2)) * matchOthersAB[t]) + (((l/4) * (r**2)) * matchOthersAA[t])
        V_AB += ((l/4) * (r2**2) * matchOthersAB[t]) + ((l/4) * (r2**2) * matchOthersAA[t])

        V_BB = (nSource * bgNoiseBA) + (nSource * bgNoiseBB) + (n2 * bgNoiseBC) + (mSource * contextNoiseBA) + (mSource * contextNoiseBB) + (m2 * contextNoiseBC)
        V_BB += ((r ** 2) * selfMatchBB[t]) + (((l/4 - 1) * (r ** 2)) * matchOthersBB[t]) + (((l/4) * (r**2)) * matchOthersBA[t])
        V_BB += ((l/4) * (r2**2) * matchOthersBB[t]) + ((l/4) * (r2**2) * matchOthersBA[t])

        V_A0 = (nSource * bgNoiseAA) + (nSource * bgNoiseAB) + (n2 * bgNoiseAC) + (mSource * contextNoiseAA) + (mSource * contextNoiseAB) + (m2 * contextNoiseAC)
        V_A0 += ((l/4 * (r ** 2)) * matchOthersAA[t]) + (((l/4) * (r**2)) * matchOthersAB[t])
        V_A0 += ((l/4) * (r2**2) * matchOthersAA[t]) + ((l/4) * (r2**2) * matchOthersAB[t])

        V_B0 = (nSource * bgNoiseBA) + (nSource * bgNoiseBB) + (n2 * bgNoiseBC) + (mSource * contextNoiseBA) + (mSource * contextNoiseBB) + (m2 * contextNoiseBC)
        V_B0 += ((l/4 * (r ** 2)) * matchOthersBB[t]) + (((l/4) * (r**2)) * matchOthersBA[t])
        V_B0 += ((l/4) * (r2**2) * matchOthersBB[t]) + ((l/4) * (r2**2) * matchOthersBA[t])

        V_AAstrong = (nSource * bgNoiseAA) + (nSource * bgNoiseAB) + (n2 * bgNoiseAC) + (mSource * contextNoiseAA) + (mSource * contextNoiseAB) + (m2 * contextNoiseAC)
        V_AAstrong += ((r2 ** 2) * selfMatchAA[t]) + (((l/4 - 1) * (r2 ** 2)) * matchOthersAA[t]) + (((l/4) * (r2**2)) * matchOthersAB[t])
        V_AAstrong += ((l/4) * (r**2) * matchOthersAA[t]) + ((l/4) * (r**2) * matchOthersAB[t])

        V_BAstrong = (nSource * bgNoiseBA) + (nSource * bgNoiseBB) + (n2 * bgNoiseBC) + (mSource * contextNoiseBA) + (mSource * contextNoiseBB) + (m2 * contextNoiseBC)
        V_BAstrong += ((r2 ** 2) * selfMatchBA[t]) + (((l/4 - 1) * (r2 ** 2)) * matchOthersBA[t]) + (((l/4) * (r2**2)) * matchOthersBB[t])
        V_BAstrong += ((l/4) * (r**2) * matchOthersBA[t]) + ((l/4) * (r**2) * matchOthersBB[t])

        V_ABstrong = (nSource * bgNoiseAA) + (nSource * bgNoiseAB) + (n2 * bgNoiseAC) + (mSource * contextNoiseAA) + (mSource * contextNoiseAB) + (m2 * contextNoiseAC)
        V_ABstrong += ((r2 ** 2) * selfMatchAB[t]) + (((l/4 - 1) * (r2 ** 2)) * matchOthersAB[t]) + (((l/4) * (r2**2)) * matchOthersAA[t])
        V_ABstrong += ((l/4) * (r**2) * matchOthersAB[t]) + ((l/4) * (r**2) * matchOthersAA[t])

        V_BBstrong = (nSource * bgNoiseBA) + (nSource * bgNoiseBB) + (n2 * bgNoiseBC) + (mSource * contextNoiseBA) + (mSource * contextNoiseBB) + (m2 * contextNoiseBC)
        V_BBstrong += ((r2 ** 2) * selfMatchBB[t]) + (((l/4 - 1) * (r2 ** 2)) * matchOthersBB[t]) + (((l/4) * (r2**2)) * matchOthersBA[t])
        V_BBstrong += ((l/4) * (r**2) * matchOthersBB[t]) + ((l/4) * (r**2) * matchOthersBA[t])

        #difference between the sources

        S_A = np.sqrt(V_AA + V_AB)
        S_B = np.sqrt(V_BB + V_AB)
        S_AStrong = np.sqrt(V_AAstrong + V_BAstrong)
        S_BStrong = np.sqrt(V_BBstrong + V_ABstrong)
        S_A0 = np.sqrt(V_A0 + V_B0)
        S_B0 = np.sqrt(V_A0 + V_B0)
        SAvgWeak = np.mean(np.vstack((S_A, S_B)), axis = 0)
        SAvgStrong = np.mean(np.vstack((S_AStrong, S_BStrong)), axis = 0)
        SAvg = np.mean(np.vstack((S_A, S_B, S_AStrong, S_BStrong)), axis = 0)
           
        d = (UAAvg[t] + UBAvg[t])/SAvg
        dWeak = (UA[t] + UB[t])/SAvgWeak
        dStrong = (UAStrong[t] + UBStrong[t])/SAvgStrong
        d0 = 0

        #just going to use the equal variance version here for now - can use an unequal variance version with unequal source strength     
        if fullyInformed == False:

          muA = dWeak * d/2
          muAStrong = dStrong * d/2
          muB = -(dWeak * d/2)
          muBStrong = -(dStrong * d/2)
          mu0A = d0 * d/2
          mu0B = -(d0 * d/2)

          #variance is solely due to expected strength! so for an uninformed model the expected variance is the same for both weak and strong items
          sigma = d
          sigmaStrong = d
          sigmaLure = d
        #below is the fully informed model = the case here is that subjects could use item strength to estimate the source strength
        else:
          muA = dWeak**2/2
          muAStrong = dStrong**2/2
          muB = -(dWeak**2)/2
          muBStrong = -(dStrong**2)/2

          d0 = dWeak #just assume that they use the d for weak items
          mu0A = d0 * d/2
          mu0B = -(d0 * d/2)

          sigma = dWeak
          sigmaStrong = dStrong
          sigmaLure = d0

        if oldnew == 'weakA': mu, sigma = muA, sigma
        elif oldnew == 'weakB': mu, sigma = muB, sigma
        elif oldnew == 'strongA': mu, sigma = muAStrong, sigmaStrong
        elif oldnew == 'strongB': mu, sigma = muBStrong, sigmaStrong
        elif oldnew == 'lure':  mu, sigma = mu0A, sigmaLure

        musSource[wf_idx & cond_idx & type_idx], sigmasSource[wf_idx & cond_idx & type_idx] = mu, sigma

  cdfs = norm.cdf(theta[:,p.index('crit')].reshape(1,nChains), loc = mus, scale = sigmas)
  cdfsSource = norm.cdf(theta[:,p.index('critSource')].reshape(1,nChains), loc = musSource, scale = sigmasSource)

  if pred == True:
    modelRates = np.zeros((7,3,2,nChains))
    counts = np.zeros((5,3,2,3,1)) #type (lure, sourceAweak, sourceBweak, sourceAstrong, sourceBstrong) x cond x wf
    for w, wf in enumerate(['HF', 'LF']):
      wf_idx = data['wf'] == wf

      for c, cond in enumerate(['pure', 'mixed', 'long']):
        cond_idx = data['cond'] == cond

        modelRates[4,c,w] = 1.0 - np.mean(cdfs[wf_idx & cond_idx & (data['type'] == 'lure')], axis = 0)
        modelRates[5,c,w] = 1.0 - np.mean(cdfs[wf_idx & cond_idx & (data['type'] == 'weak')], axis = 0)
        modelRates[6,c,w] = 1.0 - np.mean(cdfs[wf_idx & cond_idx & (data['type'] == 'strong')], axis = 0)

        modelRates[0,c,w] = 1.0 - np.mean(cdfsSource[wf_idx & cond_idx & (data['type'] == 'weak') & (data['source'] == 0)], axis = 0)
        modelRates[1,c,w] = np.mean(cdfsSource[wf_idx & cond_idx & (data['type'] == 'weak') & (data['source'] == 1)], axis = 0)

        modelRates[2,c,w] = 1.0 - np.mean(cdfsSource[wf_idx & cond_idx & (data['type'] == 'strong') & (data['source'] == 0)], axis = 0)
        modelRates[3,c,w] = np.mean(cdfsSource[wf_idx & cond_idx & (data['type'] == 'strong') & (data['source'] == 1)], axis = 0)

        for i in xrange(3):
          counts[4,c,w,i] = np.sum(data[wf_idx & cond_idx & (data['type'] == 'lure')]['response'] == i)
          counts[0,c,w,i] = np.sum(data[wf_idx & cond_idx & (data['type'] == 'weak') & (data['source'] == 0)]['response'] == i)
          counts[1,c,w,i] = np.sum(data[wf_idx & cond_idx & (data['type'] == 'weak') & (data['source'] == 1)]['response'] == i)

          if cond == 'mixed':
            counts[2,c,w,i] = np.sum(data[wf_idx & cond_idx & (data['type'] == 'strong') & (data['source'] == 0)]['response'] == i)
            counts[3,c,w,i] = np.sum(data[wf_idx & cond_idx & (data['type'] == 'strong') & (data['source'] == 1)]['response'] == i)
    
    counts = counts.reshape(5,3,2,3) #get rid of the last dimension (length 1)
    dataRates = np.zeros((7,3,2))

    sm = np.sum(counts[0:1,:,:,0:2], axis = 3).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[0] = counts[0,:,:,0] / sm

    sm = np.sum(counts[1:2,:,:,0:2], axis = 3).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[1] = counts[1,:,:,1] / sm

    sm = np.sum(counts[2:3,:,:,0:2], axis = 3).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[2] = counts[2,:,:,0] / sm

    sm = np.sum(counts[3:4,:,:,0:2], axis = 3).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[3] = counts[3,:,:,1] / sm

    sm = np.sum(counts[4:5], axis = 3).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[4] = (counts[4,:,:,0] + counts[4,:,:,1]) / sm #FAR = (source A + source B) / CR

    #weird indexing below... if you use [0,1] for the first dimension, subsequent indices are off...
    sm = np.sum(counts[0:2], axis = (0,3)).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[5] = np.sum(counts[0:2,:,:,[0,1]], axis = (0,3)) / sm

    sm = np.sum(counts[2:4], axis = (0,3)).astype(np.float32)
    sm[sm == 0] = 1
    dataRates[6] = np.sum(counts[2:4,:,:,[0,1]], axis = (0,3)) / sm
    
    return modelRates, dataRates
    
  elif pred == False:

    cdfs = norm.cdf(theta[:,p.index('crit')].reshape(1,nChains), loc = mus, scale = sigmas)
    cdfsSource = norm.cdf(theta[:,p.index('critSource')].reshape(1,nChains), loc = musSource, scale = sigmasSource)
    #convert to prs
    prs = np.zeros((3,len(data),nChains))

    rate = 1 - cdfs

    #sourceA/sourceB judgments are weighted by the HR (source judgment is given only if it's recognized)
    prs[0] = rate * (1 - cdfsSource)
    prs[1] = rate * (cdfsSource)
    prs[2] = 1 - rate

    #print 'Miss', np.round(np.mean(prs[2,weak]), 3), np.round(np.mean(counts[2,weak]), 3), 'CR', np.round(np.mean(prs[2,lure]), 3), np.round(np.mean(counts[2,lure]), 3)

    #make sure that no values are zero
    prs = ((prs * 1000000) + .5) / 1000001

    #print np.where(np.isnan(prs))


    d = np.ones((len(data),nChains))
    for i in xrange(len(data)):
      if simulate == False:
        m = Multinomial(prs[:,i])
        d[i] = m.pmf(counts[:,i].reshape(3,1))
      elif simulate == True:
        for chain in xrange(nChains):
          s = np.random.multinomial(1, prs[:,i,chain])
          if s[0] == 1: resp = 0
          elif s[1] == 1: resp = 1
          else: resp = 2
          d[i,chain] = resp  
         
    if (full == False) & (simulate == False):
      d = np.sum(np.log(d), axis = 0)
    elif simulate == False:
      d = np.swapaxes(d,0,1) #swap chains and conditions
    return d


def runWAIC(directory, dataset, thin = 10, prior = False):
  print 'Running WAIC for ' + str(directory)
  data = returnData(dataset)
  nSubj = len(np.unique(data['subj']))

  log_dens_like = pickle.load(open('pickles/' + directory + '/log_dens_like.pkl', 'rb'))
  params = getParams(directory)

  theta0 = getSubj(0, directory)
  nChains, nmc, nParams = theta0.shape
  theta = np.zeros((nSubj,nChains,nmc,nParams))
  theta[0] = theta0
  for subj in xrange(1, nSubj): theta[subj] = getSubj(subj, directory)
  print 'Subjects collected...'

  if prior == False:
    lpd, pwaic, waic, complete = de.calculateWAIC2(data, params, log_dens_like, theta, thin = thin, nJobs = -1)
    title = 'WAIC.pkl'
    title2 = 'WAICsubj.pkl'
  elif prior == True:
    lpd, pwaic, waic, complete = de.calculateWAIC_prior(data, params, getDist(directory), log_dens_like, theta, getPhiMu(directory), getPhiSigma(directory), thin = thin, nJobs = -1)
    title = 'WAICprior.pkl'
    title2 = 'WAICpriorsubj.pkl'
  pickle.dump((lpd, pwaic, waic), open('pickles/' + directory + '/' + title, 'wb'))
  pickle.dump(complete, open('pickles/' + directory + '/' + title2, 'wb'))

def runWAICBetween(directory, dataset, thin = 10):

  ps = getps(directory)
  params = getParams(directory)
  dataset = dataName(directory)
  log_dens_like = getLogDensLike(directory)
  data = returnData(dataset)
  nSubj = len(np.unique(data['subj']))
  nSubjs = [len(np.unique(data[data['exp'] == exp]['subj'])) for exp in [1,2,3]]
  
  #assemble thetas
  thetas = []
  c = 0
  for t in xrange(3):
    theta0 = getSubj(c, directory)
    theta = np.zeros((nSubjs[t], theta0.shape[0], theta0.shape[1], theta0.shape[2]))
    for i in xrange(nSubjs[t]):
      theta[i] = getSubj(i+c, directory)
    thetas.append(theta)
    c += nSubjs[t]

    x = 0
  print 'Subjects collected!'
    
  lpd, pwaic, waic, complete = de.calculateWAIC2Between(data, params, log_dens_like, thetas, ps, nSubjs, thin = thin, nJobs = -1)
  title = 'WAIC.pkl'
  title2 = 'WAICsubj.pkl'
  pickle.dump((lpd, pwaic, waic), open('pickles/' + directory + '/' + title, 'wb'))
  pickle.dump(complete, open('pickles/' + directory + '/' + title2, 'wb'))

    
def synthDataSourceDev(fit = 'sourceDev4COCO', thin = 20, nJobs = -1):

  dataset = dataName(fit)
  logDensLike = getLogDensLike(fit)

  sdCut = False

  data = returnData(dataset, sdCut = sdCut)
  subjectList = np.unique(data['subj'])
  nSubj = len(subjectList)
  p = getParams(fit)

  #collect subjects
  theta0 = getSubj(0, fit)
  nChains, nmc, nParams = theta0.shape
  nSamples = nmc / thin
  theta = np.zeros((nSubj,nChains,nmc,nParams))
  theta[0] = theta0
  for subj in xrange(1, nSubj): theta[subj] = getSubj(subj, fit)
  print 'Subjects collected... Synthesizing data...'

  theta = theta[:,:,np.arange(0,nmc,thin)]
  
  rates = np.zeros((nSubj,7,3,2))
  modelRates = np.zeros((nSubj,7,3,2,nChains,nSamples))

  for i, subject in enumerate(np.unique(data['subj'])):
    for j in xrange(nSamples):
      modelRates[i,:,:,:,:,j], rates[i] = logDensLike(theta[i,:,j], data[data['subj'] == subjectList[i]], p, pred = True)
  
  modelRates = modelRates.reshape(nSubj,7,3,2,nChains*nSamples)
  rates = np.ma.masked_array(rates)
  pickle.dump([rates, modelRates], open('pickles/' + fit + '/avgPred.pkl', 'wb'))

def genKdePriors(fit, dataset = 'sourceLSE1', gridsize = 5000):

  from statsmodels.nonparametric.kde import KDEUnivariate as KDE
  from scipy.stats import gaussian_kde
  phiMu = getPhiMu(fit)
  phiSigma = getPhiSigma(fit)
  p = getParams(fit)

  kdePriorMu, kdePriorSigma = {}, {}
    
  #exclude criterion parameters
  paramList = [param for param in p if 'crit' not in param]

  print '----'
  print 'Generating KDE priors for ' + str(fit) + ' for fitting dataset ' + str(dataset)
  print '----'

  for param in paramList:
    if 'log' in param: lower, upper = -1 * np.inf, np.inf
    else: lower, upper = 0, np.inf

    #I previously did this with the statsmodels function - however, I needed to customize the grid size

    vals = phiMu[:,:,p.index(param)].flatten()
    min = np.max([lower, np.min(vals) - (1.5 * np.std(vals))])
    max = np.min([upper, np.max(vals) + (1.5 * np.std(vals))])
    support = np.linspace(min, max, gridsize)
    kde = gaussian_kde(vals)
    density = kde(support)

    mu = {}
    mu['support'] = support
    mu['density'] = density

    upper = np.inf
    vals = phiSigma[:,:,p.index(param)].flatten()
    min = np.max([lower, np.min(vals) - np.std(vals)])
    max = np.min([upper, np.max(vals) + np.std(vals)])
    support = np.linspace(min, max, gridsize)
    kde = gaussian_kde(vals)
    density = kde(support)
  
    sigma = {}
    sigma['support'] = support
    sigma['density'] = density

    kdePriorMu[param] = mu
    kdePriorSigma[param] = sigma 

  kdePriors = {}
  kdePriors['mu'] = kdePriorMu
  kdePriors['sigma'] = kdePriorSigma
      
  pickle.dump(kdePriors, open('pickles/' + fit + '/kdePriors.pkl', 'wb'))
  print 'Finished with KDE!'

### MODEL RECOVERY

#generate data for recovery
def genData(fit, dataset = -1, version = '', **kwargs):
  #get factors and relevant cutoffs for each dataset
  if dataset == -1: dataset = dataName(fit)
  data = returnData(dataset)

  if 'sourceDev' in dataset: factors = ['cond', 'source', 'ageFreq', 'trial', 'age', 'wf', 'type']

  theta0 = getSubj(0, fit)
  nChains, nmc, nParams = theta0.shape
  nSubj = len(np.unique(data['subj']))
  theta = np.zeros((nSubj,nChains,nmc,nParams))
  theta[0] = theta0
  for subj in xrange(1, nSubj): theta[subj] = getSubj(subj, fit)
  print 'Subjects collected'

  log_dens_like = getLogDensLike(fit)
  p = getParams(fit)

  genData, th = de.genRecovery(data, p, log_dens_like, theta, nJobs = -1, factors = factors, respRT = False, respField = 'response')
  np.save('pickles/' + fit + '/gen_th.npy', th)
  
  np.save('datasets/' + dataset + 'Rec' + version + '.npy', genData)
  
  print 'Generated predictions!'

def runFit(directory = '', nChains = -1, nmc = 1500, burnin = 0, thin = 1, nJobs = -1, fit = 'hier', DIC = True, WAIC = True, synthData = False, informedStart = -1, informedPrior = None, between = False, **kwargs):
  dataset = dataName(directory)
  data = returnData(dataset)
  if informedStart == True: informedStart = directory

  if dataset == 'length':
    data = pickle.load(open('source.pkl', 'rb'))
    data = data[data['task'] == 'IR']
    data = data[(data['RT'] > .2) & (data['RT'] < 2.5)]
    params, dist = paramNamesLength(data)
    log_dens_like = logDensLikeLength
  elif dataset == 'conc':

    if 'rAssoc' in kwargs.keys(): rAssoc = kwargs['rAssoc']
    else: rAssoc = False
    if 'UttWF' in kwargs.keys(): UttWF = kwargs['UttWF']
    else: UttWF = True
    if 'VttWF' in kwargs.keys(): VttWF = kwargs['VttWF']
    else: VttWF = True
    if 'VttConc' in kwargs.keys(): VttConc = kwargs['VttConc']
    else: VttConc = True

    data = returnData('conc', exclude = False)
    params, dist = paramNamesConc(data, UttWF = UttWF, rAssoc = rAssoc, VttWF = VttWF, VttConc = VttConc)
    log_dens_like = logDensLikeConc
    lower, upper = 0, np.inf
  elif ('source' in dataset):
    mixture = kwargs.get('mixture', False)
    if 'Mix' in directory: mixture = True
    if directory == 'sourceLSE2': mixture = False #automatically disable the mixture in Experiment 2
    
    Uaa = kwargs.get('Uaa', False)
    if 'Uaa' in directory: Uaa = True

    if 'sourceProp' in kwargs.keys(): sourceProp = kwargs['sourceProp']
    else: sourceProp = False
    if 'prop' in directory: sourceProp = True

    if 'CN' in kwargs.keys(): CN = kwargs['CN']
    else: CN = False
    if 'CN' in directory: CN = True

    critMix = kwargs.get('critMix', False)
    Vcc = kwargs.get('Vcc', False)

    rWeak = kwargs.get('rWeak', True)
    m = kwargs.get('m', False)
    Vti = kwargs.get('Vti', True)
    Vsu = kwargs.get('Vsu', False)
    UssSource = kwargs.get('Uss', False)
    critMixture = kwargs.get('critMixture', True)
    critGuess = kwargs.get('critGuess', False)
    fullyInformed = kwargs.get('fullyInformed', False)
    if 'FI' in fit: fullyInformed = True

    freq = kwargs.get('freq', True)

    if (dataset in ['sourceLSE2', 'sourceLSE3']) & ('SDT' not in directory):
      if 'kdePriors.pkl' in os.listdir('pickles/' + informedPrior + '/'): kdePriors = pickle.load(open('pickles/' + informedPrior + '/kdePriors.pkl', 'rb'))
      else:
        genKdePriors(informedPrior)
        kdePriors = pickle.load(open('pickles/' + informedPrior + '/kdePriors.pkl', 'rb'))
      kwargs['kdePriors'] = kdePriors
      informed = True


    else: informed = False

    if between == False:
      if 'sourceLSE' in dataset:
        params, dist = paramNamesSourceLSE_bgNoise(data, critMix = critMix, Uaa = Uaa, informed = informed, Vcc = Vcc, rWeak = rWeak, m = m, Vti = Vti, Vsu = Vsu, UssSource = UssSource, critGuess = critGuess, VaaVab = VaaVab, logr = logr)
        log_dens_like = partial(logDensLikeSourceLSE_bgNoise, mixture = mixture, critMixture = critMixture, fullyInformed = fullyInformed)
      elif ('sourceDev' in dataset) & (freq == False):
        params, dist = paramNamesSourceDevelopment(data, Uaa = Uaa, informed = informed, Vcc = Vcc, rWeak = rWeak, m = m, Vti = Vti, Vsu = Vsu, UssSource = UssSource, VaaVab = VaaVab, logr = logr)
        log_dens_like = partial(logDensLikeSourceDevelopment, mixture = mixture, critMixture = critMixture, fullyInformed = fullyInformed)
      elif ('sourceDev' in dataset) & (freq == True):
        params, dist = paramNamesSourceDevelopmentFreq(data, **kwargs)
        log_dens_like = partial(logDensLikeSourceDevelopmentFreq, mixture = mixture, critMixture = critMixture, fullyInformed = fullyInformed)
    elif between == True:
      ps, params, dists, distHiers, dist = paramNamesSourceLSE_bgNoiseBetween(data, critMix = critMix, Uaa = Uaa, Vcc = Vcc, rWeak = rWeak, m = m, Vti = Vti, Vsu = Vsu, UssSource = UssSource, critGuess = critGuess, VaaVab = VaaVab, logr = logr, Vtt = Vtt, prop = prop)
      log_dens_like = partial(logDensLikeSourceLSE_bgNoise, mixture = mixture, critMixture = critMixture, fullyInformed = fullyInformed)

      exps = np.unique(data['exp'])
      nSubjs = [len(np.unique(data[data['exp'] == exp]['subj'])) for exp in exps]

  #SDT fit for unrecognized items
  if ('sourceLSE3' in dataset) & ('SDT' in directory):
    if 'nod' in kwargs.keys(): ds = False
    else: ds = True
    
    if 'nod' in directory: ds = False
    else: ds = True

    if 'recog' in directory: recog = True
    else: recog = False

    if 'nols' in kwargs.keys(): d_ls = False
    else: d_ls = True
    if 'nols' in directory: d_ls = False

    if 'nowf' in kwargs.keys(): d_wf = False
    else: d_wf = True
    if 'nowf' in directory: d_wf = False

    if 'nolswf' in kwargs.keys(): d_wf, d_ls = False, False
    if 'nolswf' in directory: d_wf, d_ls = False, False

    if 'single' in directory: singleTrial = True
    else: singleTrial = False

    if 'nomix3' in directory: mixture3 = False
    else: mixture3 = True

    freq = False
    
    itemRating = 3
    if 'item4' in directory: itemRating = 4
    elif 'item5' in directory: itemRating = 5

    params, dist = paramNamesSourceUnrecognized(data, ds = ds, crit_ls = True, d_ls = d_ls, d_wf = d_wf)
    log_dens_like = partial(logDensLikeSourceLSEUnrecognized, itemRating = itemRating, recog = recog, singleTrial = singleTrial, mixture3 = mixture3)

  #create directory if it doesn't alraedy exist
  if directory != '':
    if not os.path.exists(os.getcwd() + '/pickles/' + directory):
      print 'Directory not in pickles folder! Creating...'
      os.makedirs(os.getcwd() + '/pickles/' + directory)
    
  if nChains == -1: nChains = 3 * len(params)
  elif nChains == -4: nChains = 4 * len(params)
  
  if directory != '': directory += '/'
  print directory

  priors = generateHyperPriors(params, dist, params)
  #get the informed priors for the Criss dataset

  k = priors.keys()
  k.sort()
  for prm in k:
    if '_mumu' in prm: print prm + ': ' + str(priors[prm])
    if '_musigma' in prm: print prm + ': ' + str(priors[prm])
  starts = startPoints(params)
  
  print starts

  rp = .001

  if between == False:
    theta, phiMu, phiSigma, like, weight, hyperWeight, priors = de.samplingHier(data, params, dist, log_dens_like, starts, priors, nChains, nmc, burnin, thin = thin, nJobs = nJobs, informedStart = informedStart, rp = rp, **kwargs)
    for subj in xrange(theta.shape[0]): np.save('pickles/' + directory + 'theta_s' + str(subj) + '.npy', theta[subj])
  elif between == True:
    thetas, phiMu, phiSigma, like, weight, hyperWeight, priors = de.samplingHierBetween(data, ps, params, dists, distHiers, dist, nSubjs, log_dens_like, starts, priors, nChains, nmc, burnin, thin = thin, nJobs = nJobs, informedStart = informedStart, rp = rp, **kwargs)
    x = 0
    for theta in thetas:
      print theta.shape[0]
      for subj in xrange(theta.shape[0]): np.save('pickles/' + directory + 'theta_s' + str(subj + x) + '.npy', theta[subj])
      x += theta.shape[0]
    pickle.dump(dists, open('pickles/' + directory + 'dists.pkl', 'wb'))
    pickle.dump(distHiers, open('pickles/' + directory + 'distHiers.pkl', 'wb'))
    pickle.dump(ps, open('pickles/' + directory + 'ps.pkl', 'wb'))

  np.save('pickles/' + directory + 'phiMus.npy', phiMu)
  np.save('pickles/' + directory + 'phiSigmas.npy', phiSigma)
  np.save('pickles/' + directory + 'likes.npy', like)
  np.save('pickles/' + directory + 'weights.npy', weight)
  np.save('pickles/' + directory + 'hyperWeight.npy', hyperWeight)
  pickle.dump(params, open('pickles/' + directory + 'params.pkl', 'wb'))
  pickle.dump(dist, open('pickles/' + directory + 'dist.pkl', 'wb'))
  pickle.dump(log_dens_like, open('pickles/' + directory + 'log_dens_like.pkl', 'wb'))
  pickle.dump(priors, open('pickles/' + directory + 'priors.pkl', 'wb'))

  #check convergence
  subjectList = np.unique(data['subj'])
  if between == False:
    for param in params:
      for subj, subject in enumerate(subjectList):
        gr = gelman_rubin(theta[subj,:,:,params.index(param)])
        if gr > 1.10: print 'Subject: ' + str(subj) + ' Parameter: ' + param + ' GR: ' + str(np.around(gr, 4))
  else:
    x = 0
    for g in xrange(len(thetas)):
      theta, p = thetas[g], ps[g]
      for param in p:
        for subj in xrange(theta.shape[0]):
          gr = gelman_rubin(theta[subj,:,:,p.index(param)])
          if gr > 1.10: print 'Subject: ' + str(subj + x) + ' Parameter: ' + param + ' GR: ' + str(np.around(gr, 4))
      x += theta.shape[0]

    for param in params:
      gr = gelman_rubin(phiMu[:,:,params.index(param)])
      if gr > 1.10: print 'phiMu Parameter: ' + param + ' GR: ' + str(np.around(gr, 4))
      gr = gelman_rubin(phiSigma[:,:,params.index(param)])
      if gr > 1.10: print 'phiSigma Parameter: ' + param + ' GR: ' + str(np.around(gr, 4))

  if DIC == True:
    DIC = de.calculateDIC(data, params, log_dens_like, theta, like)
    pickle.dump(DIC, open('pickles/' + directory + 'DIC.pkl', 'wb'))
  if WAIC == True:
    if 'WAICthin' in kwargs.keys(): WAICthin = kwargs['WAICthin']
    else: WAICthin = 5
    if between == False:
      if freq == False: lpd, pwaic, waic, complete = de.calculateWAIC2(data, params, log_dens_like, theta, thin = WAICthin, nJobs = nJobs)
      else: lpd, pwaic, waic, complete = de.calculateWAIC(data, params, log_dens_like, theta, thin = WAICthin, nJobs = nJobs)
    else: lpd, pwaic, waic, complete = de.calculateWAIC2Between(data, params, log_dens_like, thetas, ps, nSubjs, thin = WAICthin, nJobs = nJobs)
    print 'WAIC: ' + str(waic)
    title = 'WAIC.pkl'
    title2 = 'WAICsubj.pkl'
    pickle.dump((lpd, pwaic, waic), open('pickles/' + directory + '/WAIC.pkl', 'wb'))
    pickle.dump(complete, open('pickles/' + directory + '/WAICsubj.pkl', 'wb'))
  if synthData == True:
    print 'Synthesizing data...'
    subjs, summaries = de.synthData(data, params, log_dens_like, theta, burnin = nmc / 3, factors = factors)
    for subj in xrange(len(subjs)): pickle.dump(subjs[subj], open('pickles/' + directory + 'synthData_s' + str(subj) + '.pkl', 'wb'))
    for summary in summaries.keys(): pickle.dump(summaries[summary], open('pickles/' + directory + summary + '.pkl', 'wb'))

def main():
  ### setings for DE-MCMC
  mStart, mStop = 3000, 3400
  migration = 40
  burnin = 25000
  nmc = 10000
  thin = 20
  nJobs = -1 #using all possible CPU cores

  ## fit Adults
  runFit('sourceDevAdultsCOCOfreqVaa2', burnin = burnin, nmc = nmc, thin = thin, gamma1 = True, WAIC = True, freq = True, nJobs = nJobs, logr = False, Vac = True)
  synthDataSourceDev('sourceDevAdultsCOCOfreqVaa2')
  ## fit 78yrs
  runFit('sourceDev78COCOfreqVaa2', burnin = burnin, nmc = nmc, thin = thin, gamma1 = True, WAIC = True, freq = True, nJobs = nJobs, logr = False, Vac = True)
  synthDataSourceDev('sourceDev78COCOfreqVaa2')
  ## fit 45yrs
  runFit('sourceDev45COCOfreqVaa2', burnin = burnin, nmc = nmc, thin = thin, gamma1 = True, WAIC = True, freq = True, nJobs = nJobs, logr = False, Vac = True)
  synthDataSourceDev('sourceDev45COCOfreqVaa2')  
  ## plot
  plotSourceDev.savePlots('Vaa2')


if __name__ == "__main__":
    main()

