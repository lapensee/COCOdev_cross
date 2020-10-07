import cPickle as pickle
import numpy as np
import os
#import de
from scipy.stats import norm, zscore
import difflib

#updated, now uses .npy files instead of .pkl files for more space efficiency

### returning datasets ###

def exclusion(dataset, lower, upper, cutoff = .2, acc = False):
  data = returnData(dataset, exclude = False)
  cut = []
  corrs = []
  for subj, subject, in enumerate(np.unique(data['subj'])):
    if np.mean((data[data['subj'] == subject]['RT'] < lower) | (data[data['subj'] == subject]['RT'] > upper)) > cutoff:
      cut.append(subject)
      corrs.append(np.mean(data[data['subj'] == subject]['correct']))
  if acc == True: print corrs
  return cut

def get_d(dataset):
  target, lure = data['type'] == 'target', data['type'] == 'lure'
  hits = np.array([np.sum(data[target & (data['subj'] == subject)]['response']) for subject in np.unique(data['subj'])])
  nTargets = np.array([len(data[target & (data['subj'] == subject)]['response']) for subject in np.unique(data['subj'])])
  FA = np.array([np.sum(data[lure & (data['subj'] == subject)]['response']) for subject in np.unique(data['subj'])])
  nLures = np.array([len(data[lure & (data['subj'] == subject)]['response']) for subject in np.unique(data['subj'])])
  d = norm.ppf((hits + .5) / (nTargets + 1)) - norm.ppf((FA + .5) / (nLures + 1))
  return d

def returnData(dataset, exclude = True, excludeSubj = False, sdCut = False, pkl = False):
  if 'srw' in dataset: lower, upper = .2, 2.5
  if dataset == 'srw': data = pickle.load(open('datasets/SRW_E1within.pkl', 'rb'))
  elif dataset == 'srw2': data = pickle.load(open('datasets/SRW_E1mixed.pkl', 'rb'))
  elif dataset == 'srw3': data = pickle.load(open('datasets/SRW_E2within.pkl', 'rb'))
  elif dataset == 'srw4': data = pickle.load(open('datasets/SRW_E2mixed.pkl', 'rb'))
  elif dataset == 'srw5': data = pickle.load(open('datasets/SRW_E3mixed.pkl', 'rb')) 
  elif dataset == 'srwmx': data = pickle.load(open('datasets/SRW_mixedpooled.pkl', 'rb'))
  elif dataset == 'srwp': data = pickle.load(open('datasets/SRW_purepooled.pkl', 'rb'))
  elif 'srm' in dataset:
    if ('SRM.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/SRM.npy')
    else:
      data = pickle.load(open('datasets/SRM.pkl', 'rb'))
    if dataset == 'srmspeed': data = data[data['sa'] == 'speed']
    elif dataset == 'srmaccuracy': data = data[data['sa'] == 'accuracy']
    lower, upper = .25, 3.0

  elif ('S14_2' in dataset) | ('SRecovery' in dataset):
    if dataset + '.npy' in os.listdir('datasets'): data = np.load('datasets/' + dataset + '.npy')
    elif dataset + '.pkl' in os.listdir('datasets'): data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    lower, upper = .2, 2.50
  elif dataset in ['seqCon', 'seqConBEAGLE', 'seqConBEAGLEcelex']:
    if (dataset + '.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/' + dataset + '.npy')
    else:
      data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    #data = data[(data['conf'] == 'def') | (data['conf'] == 'prob')]
    data = data[data['wf'] != 'NA']
    data = data[data['resp'] != -1]
    if 'celex' in dataset: data = data[np.isfinite(data['celex'])]
    lower, upper = .2, 2.5
  elif dataset in ['rae', 'raeBEAGLE', 'raeSemSpace', 'raeBEAGLEcelex']:
    if (dataset + '.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/' + dataset + '.npy')
    else:
      data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    data = data[data['wf'] != 'NA']
    data = data[data['resp'] != -1]
    if 'celex' in dataset: data = data[np.isfinite(data['celex'])]
    lower, upper = .2, 2.5
  elif dataset == 'sim2':
    data = pickle.load(open('datasets/sim2.pkl', 'rb'))
    lower, upper = .15, 3.0
  elif dataset == 'length':
    data = pickle.load(open('datasets/source.pkl', 'rb'))
    data = data[data['task'] == 'IR']
  elif 'AnnisSD' in dataset:
    if (dataset + '.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/' + dataset + '.npy')
    else:
      data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    lower, upper = .25, 10
  elif 'DR' in dataset:
    data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    lower, upper = .25, 4.0
  elif 'BS' in dataset:
    data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    lower, upper = .25, 3.0
  elif 'Dube' in dataset:
    if dataset == 'Dube1': data = pickle.load(open('datasets/Dube1.pkl', 'rb'))
    elif dataset == 'Dube2': data = pickle.load(open('datasets/Dube2.pkl', 'rb'))
    lower, upper = .25, 3.0
  elif (dataset in ['oi', 'oi0', 'oi0_b0']) | ('oiRecovery' in dataset): #oi0 is the same dataset, but just the first block!
    if (dataset + '.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/' + dataset + '.npy')
    else:
      data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    lower, upper = .25, 3.0
    if exclude == True: data = data[data['trial'] != 0]
  elif dataset == 'WPpo':
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .25, 3.0
  elif dataset == 'WPrule':
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .25, 3.0
  elif (dataset == 'C') | ('CRecovery' in dataset) | (dataset == 'C10'): #this is Experiment 2 of Criss (2010): WF and strength manipulated
    if dataset in ['C', 'C10']:
      if 'C10.npy' in os.listdir('datasets'): data = np.load('datasets/C10.npy')
      elif 'C10.pkl' in os.listdir('datasets'): data = pickle.load(open('datasets/C10.pkl', 'rb'))
    elif 'CRecovery' in dataset: data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    lower, upper = .2, 2.5
  elif dataset == 'Criss2BEAGLEcelex':
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .2, 2.5
    data = data[np.isfinite(data['celex'])] #only use valid celex values!
  elif 'Criss2BEAGLE' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .2, 2.5
  elif dataset == 'Cpo': #this is Experiment 1 of Criss (2010): prob. old manipulated
    data = pickle.load(open('datasets/Cpo.pkl', 'rb'))
    lower, upper = 0, np.inf
  elif dataset in ['sc', 'sc2']:
    if dataset == 'sc':
      data = pickle.load(open('datasets/SC.pkl', 'rb'))
      if exclude == True:
        #subjects = [6,16,31,32] #exclude these subjects - VERY fast responses
        subjects = [31, 2, 12, 16, 26, 27, 30]
        for subj in subjects:
          data = data[data['subj'] != subj]
        data = data[(data['RT'] > .3) & (data['RT'] < 4)]
    elif dataset == 'sc2':
      data = pickle.load(open('SC2.pkl', 'rb')) 
      if exclude == True:
        data = data[(data['RT'] > .3) & (data['RT'] < 4)]
  elif 'conc' in dataset:
    if (dataset + '.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/' + dataset + '.npy')
    else:
      data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    if dataset == 'concIR': data = data[data['task'] == 'IR']
    elif dataset == 'concAR': data = data[data['task'] == 'AR']
    if exclude == True:
      data = data[(data['RT'] > .3) & (data['RT'] < 4.0) & ~((data['task'] == 'IR') & (data['RT'] > 3.0))] #the ~ flips the indexing
      data = data
    exclude = False #handle the exclusion in this function, because it varies by task
    lower, upper = 0, 100
  elif 'elf' in dataset:
    if (dataset + '.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/' + dataset + '.npy')
    else:
      data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    data = data[(data['resp'] == 1) | (data['resp'] == 0)] #exclude responses with 999
    lower, upper = .2, 4.0

    d = get_d(data)
    for subj, subject in enumerate(np.unique(data['subj'])):
      if d[subj] <= 0: data = data[data['subj'] != subject] #exclude subjects with d' <= 0

    if dataset == 'elf4': data = data[data['subj'] != 407]
    elif dataset == 'elf6': data = data[data['subj'] != 612]
    
  elif dataset == 'hk99_1':
    data = pickle.load(open('datasets/hk99_1first.pkl', 'rb'))
    data = data[data['serPos'] >= 0]
    lower, upper = 0, 10
  elif 'Golomb' in dataset:
    if dataset == 'GolombYoung': data = pickle.load(open('datasets/GolombYoung.pkl', 'rb'))
    elif dataset == 'GolombOlder': data = pickle.load(open('datasets/GolombOlder.pkl', 'rb'))
    else: data = np.load('datasets/' + dataset + '.npy')
    lower, upper = 0, 20
    data = data[data['serPos'] >= 0]
    if ('ISR' not in dataset) & ('Both' not in dataset): data = data[data['task'] == 'IFR']
  elif 'peers' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    data = data[data['serPos'] >= 0]
    lower, upper = 0, 20
  elif 'Howard' in dataset: #Howard et al. (2007) PBR paper
    data = pickle.load(open('datasets/Howard.pkl', 'rb'))
    data = data[data['serPos'] >= 0]
    lower, upper = 0, 20
  elif 'MO' in dataset: #Murdock and Okada 1970 dataset
    data = pickle.load(open('datasets/MO.pkl', 'rb'))
    data = data[data['serPos'] < 20]
    lower, upper = 0, 20
  elif 'KW' in dataset:
    data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    data = data[data['serPos'] >= 0]

    if exclude == True:
      #exclude subjects with mean RT below .2
      subjectList = np.unique(data['subj'])
      meanRT = np.array([np.mean(data[data['subj'] == subject]['RT']) for subject in subjectList])
      exc = subjectList[meanRT < .2]
      for subject in exc: data = data[data['subj'] != subject]

    lower, upper = 0, 20
  elif 'PNK' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    data = data[data['serPos'] >= 0]
    lower, upper = 0, 20
  elif 'U08' in dataset:
    data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    data = data[data['serPos'] >= 0]
    lower, upper = 0, 20   
  elif 'sourceLSE' in dataset:
    if (dataset + '.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/' + dataset + '.npy')
    else:
      data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    lower, upper = 0, 250
  elif 'sourceDev' in dataset:
    if (dataset + '.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/' + dataset + '.npy')
    else:
      data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    lower, upper = 0, 250
  elif 'lengthSess' in dataset:
    if (dataset + '.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/' + dataset + '.npy')
    else:
      data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    lower, upper = .2, 2.5
  elif 'collab' in dataset:
    if (dataset + '.npy' in os.listdir('datasets/')) & (pkl == False):
      data = np.load('datasets/' + dataset + '.npy')
    else:
      data = pickle.load(open('datasets/' + dataset + '.pkl', 'rb'))
    lower, upper = .2, 4.5
  elif 'testOrder' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .2, 3.0
    data = data[data['response'] >= 0] #exclude invalid responses
    if 'testOrderTCM' in dataset:
      data = data[data['trial'] >= 12]#exclude the first twelve trials if you're doing the temporal context modeling
      data = data[data['prev_resp'] != -1]
  elif 'mot' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .2, 3.0
  elif 'drt' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .2, 3.0
  elif 'BenTulLee' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .3, 4.0
    subjects = [19, 26, 33, 51, 105, 108, 111, 112, 113, 139, 159, 158, 161, 163] #subjects with 85% high confidence responses or more
    for subject in subjects: data = data[data['subj'] != subject]
  elif 'DNPower' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .2, 2.5
  elif 'Kilic' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .3, 3.0
    data = data[data['response'] >= 0] #exclude invalid responses

    #exclude some subjects with extremely slow responses
    subjects = []
    if ('Kilic1' in dataset) & (exclude == True):
      subjects = [1,27,33]
    elif ('Kilic2' in dataset) & (exclude == True):
      subjects = [26,29,30]
    elif ('Kilic3' in dataset) & (exclude == True):
      subjects = [1,4,10,12,25,28,31] #subjects at chance or a very high proportion of fast responses!
    for subject in subjects: data = data[data['subj'] != subject]
    
    if 'celex' in dataset: data = data[np.isfinite(data['celex'])] #exclude nans on celex

  elif 'Cox' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .3, 3.0
  elif 'slse' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .3, 4.0
  elif 'VanVugt' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    data = data[data['trial'] > 2]
    lower, upper = .25, 3.0
  elif 'crossPair' in dataset: #Osth and Fox associative recognition data
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .5, 4.5
  elif 'OD' in dataset: #Osth and Dennis serial recall data
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = 0, 20
    data = data[data['serPos'] >= 0] #omissions are in the data, but these are serPos == 99 ('OD6B' only!)
  elif 'FL12' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = 0, 20
    data = data[data['serPos'] >= 0] #omissions are still in the data, but these are serPos == 99
  elif 'Nosof' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .2, 2.5
  elif 'KS' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .25, 3.0
  elif 'learn' in dataset:
    data = np.load('datasets/' + dataset + '.npy')
    lower, upper = .25, 2.5

  ### EXCLUSION ###

  #exclude subjects
  if excludeSubj == True: 
    cut = exclusion(dataset, lower, upper, cutoff = .2)
    for subj in cut: data = data[data['subj'] != subj]
  #exclude trials
  if exclude == True:
    if 'RT' in data.dtype.names: data = data[(data['RT'] > lower) & (data['RT'] < upper)]
    if 'sourceDev' in dataset:
      data = data[data['trial'] < 10] #exclude second half of the long list

    if 'BEAGLE' in dataset:
      if 'contextCos' in data.dtype.names:
        f = np.isfinite(data['contextCos']) & np.isfinite(data['orderCos'])
        f = np.mean(f, axis = -1) != 0
        data = data[f]
      elif 'meanContextCos' in data.dtype.names:
        f = (np.isfinite(data['meanContextCos'])) & (np.isfinite(data['meanOrderCos']))
        data = data[f]
        measures = ['meanContextCos', 'meanOrderCos', 'maxContextCos', 'maxOrderCos', 'meanOrderP5Cos', 'maxOrderP5Cos', 'celex']

        if 'meanContextCosweak' in data.dtype.names:
          measures += ['meanContextCosweak', 'meanOrderCosweak', 'meanContextCosstrong', 'meanOrderCosstrong', 'maxContextCosweak', 'maxContextCosstrong', 'maxOrderCosweak', 'maxOrderCosstrong']

        for measure in measures:
          if measure in data.dtype.names: data[measure] = zscore(data[measure]) #convert to z-scores
    elif 'SemSpace' in dataset:
      models = ['BEAGLE', 'GLOVE', 'LSA', 'WAS']
      for model in models:
        f = np.isfinite(data['mean' + model + 'Cos'])
        data = data[f]
        data['mean' + model + 'Cos'] = zscore(data['mean' + model + 'Cos'])


  #exclude outliers greater or less than 3 SDs from the mean
  if sdCut == True:
    for subj in np.unique(data['subj']):
      meanRT, SDRT = np.mean(data[data['subj']]['RT']), np.std(data[data['subj']]['RT'])
      cutoff = (data['subj'] == subj) & ((data['RT'] > (meanRT + (3 * SDRT))) | (data['RT'] < (meanRT - (3 * SDRT))))
      data = data[cutoff == False]
  return data

#calculate d' for each subject: can use this for exclusion
def get_d(data):

  hits = np.array([np.sum(data[(data['subj'] == subject) & (data['type'] == 'target')]['resp']) for subject in np.unique(data['subj'])])
  FA = np.array([np.sum(data[(data['subj'] == subject) & (data['type'] == 'lure')]['resp']) for subject in np.unique(data['subj'])])
  nTargets = np.array([len(data[(data['subj'] == subject) & (data['type'] == 'target')]['resp']) for subject in np.unique(data['subj'])]).astype(np.float32)
  nLures = np.array([len(data[(data['subj'] == subject) & (data['type'] == 'lure')]['resp']) for subject in np.unique(data['subj'])]).astype(np.float32)

  return norm.ppf((hits + .5) / (nTargets + 1)) - norm.ppf((FA + .5) / (nLures + 1))

#this function checks the fit name and returns the dataset! Nifty!
def dataName(fit):
  datanames = os.listdir('datasets')
  for i in xrange(len(datanames)): 
    datanames[i] = datanames[i].replace('.npy', '')
    datanames[i] = datanames[i].replace('.pkl', '') #don't use strip here! strip('.pkl') will replace p's, s's, and l's!
  matches = []
  for dataname in datanames:
    if dataname in fit: matches.append(dataname)
  if len(matches) == 1: return matches[0]
  else:
    scores = np.array([difflib.SequenceMatcher(None, fit, match).ratio() for match in matches])
    return matches[scores.argmax()]

def getGenData(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'genData.pkl'))

def getPhiMu(fit = '', pkl = False):
  if fit != '': fit += '/'
  if ('phiMus.npy' in os.listdir('pickles/' + fit)) & (pkl == False):
    return np.load(open('pickles/' + fit + 'phiMus.npy', 'rb'))
  else:
    return pickle.load(open('pickles/' + fit + 'phiMus.pkl', 'rb'))

def getPhiSigma(fit = '', pkl = False):
  if fit != '': fit += '/'
  if ('phiSigmas.npy' in os.listdir('pickles/' + fit)) & (pkl == False):
    return np.load(open('pickles/' + fit + 'phiSigmas.npy'))
  else:
    return pickle.load(open('pickles/' + fit + 'phiSigmas.pkl'))

def getParams(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'params.pkl'))

def getps(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'ps.pkl'))

def getGroupParams(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'groupParams.pkl'))

def getHierParams(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'hierParams.pkl'))

def getDist(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'dist.pkl'))

def getDistHier(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'distHier.pkl'))  

def getDistGroup(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'distGroup.pkl'))  

def getPriors(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'priors.pkl'))

def getLike(fit = '', pkl = False):
  if fit != '': fit += '/'
  if ('likes.npy' in os.listdir('pickles/' + fit)) & (pkl == False):
    return np.load('pickles/' + fit + 'likes.npy')
  else: return pickle.load(open('pickles/' + fit + 'likes.pkl'))

def getWeight(fit = '', pkl = False):
  if fit != '': fit += '/'
  if ('weights.npy' in os.listdir('pickles/' + fit)) & (pkl == False):
    return np.load('pickles/' + fit + 'weights.npy')
  else: return pickle.load(open('pickles/' + fit + 'weights.pkl'))

def getGroupWeight(fit = '', pkl = False):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'groupWeight.pkl'))

def getHyperWeight(fit = '', pkl = False):
  if fit != '': fit += '/'
  if ('hyperWeight.npy' in os.listdir('pickles/' + fit)) & (pkl == False):
    return np.load('pickles/' + fit + 'hyperWeight.npy')
  else:
    return pickle.load(open('pickles/' + fit + 'hyperWeight.pkl'))

def getLogDensLike(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'log_dens_like.pkl')) 

def getSubj(subj = -1, fit = '', pkl = False):
  if fit != '': fit += '/'
  if subj != -1:
    if ('theta_s' + str(subj) + '.npy' in os.listdir('pickles/' + fit)) & (pkl == False):
      return np.load(open('pickles/' + fit + 'theta_s' + str(subj) + '.npy'))
    else:
      return pickle.load(open('pickles/' + fit + 'theta_s' + str(subj) + '.pkl'))
  else: return pickle.load(open('pickles/' + fit + 'theta.pkl'))

def getTheta(fit = '', dataset = ''):
  data = returnData(dataset)
  nSubj = len(np.unique(data['subj']))
  theta0 = getSubj(0, fit)
  nChains,nmc,nParams = theta0.shape
  theta = np.zeros((nSubj,nChains,nmc,nParams))
  theta[0] = theta0
  for i in xrange(1, nSubj):
    theta[i] = getSubj(i, fit)
  return theta

def getSynth(subj = -1, fit = '', pkl = False):
  if fit != '': fit += '/'
  if subj != -1:
    if ('synthData_s' + str(subj) + '.npy' in os.listdir('pickles/' + fit)) & (pkl == False):
      return np.load(open('pickles/' + fit + 'synthData_s' + str(subj) + '.npy'))
    else:
      return pickle.load(open('pickles/' + fit + 'synthData_s' + str(subj) + '.pkl'))
  else: return pickle.load(open('pickles/' + fit + 'synthData.pkl'))

def getDIC(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'DIC.pkl'))

def getWAIC(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'WAIC.pkl'))

def getWAICsubj(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'WAICsubj.pkl'))

def getWAICprior(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'WAICprior.pkl'))

def getDICMedian(fit = ''):
  if fit != '': fit += '/'
  return pickle.load(open('pickles/' + fit + 'DICmedian.pkl'))

def returnMedians(arr, params, burnin = 3000):
  return np.array([np.median(arr[:,burnin:,params.index(param)]) for param in params])

def returnBest(arr, weight):
  m = np.where(weight == np.max(weight))
  bestChain, bestMC = m[0][0], m[1][0]
  return np.array(arr[bestChain,bestMC])

def subjectAverage(fit = ''):
  files = os.listdir('pickles/' + fit)
  for f in files:
    nSubj = 0
    if 'theta' in f: nSubj +=1
  theta0 = getSubj(0, fit)
  nChains, nSamples, nParams = theta0.shape
  theta = np.zeros((nSubj,nChains,nSamples,nParams))
  theta[0] = theta0
  for subj in xrange(1, nSubj):
    theta[subj] = getSubj(subj, fit)
  return np.mean(theta, axis = 0)

def cut(fit, cutpoint = 1000):
  
  k = getPhiMu(fit)
  nSamples = k.shape[1]
  if (nSamples >= 0) & (cutpoint < nSamples):
    np.save('pickles/' + fit + '/phiMus.npy', k[:,cutpoint:])
    k = getPhiSigma(fit)
    np.save('pickles/' + fit + '/phiSigmas.npy', k[:,cutpoint:])
    k = getWeight(fit)
    nSubj = k.shape[0]
    np.save('pickles/' + fit + '/weights.npy', k[:,:,cutpoint:])
    k = getHyperWeight(fit)
    np.save('pickles/' + fit + '/hyperWeight.npy', k[:,cutpoint:])
    for i in xrange(nSubj):
      k = getSubj(i, fit)
      np.save('pickles/' + fit + '/theta_s' + str(i) + '.npy', k[:,cutpoint:])

def createLike(fit, dataset, nJobs = -1):
  data = returnData(dataset)
  theta = getTheta(fit, dataset)
  params = getParams(fit)
  log_dens_like = getLogDensLike(fit)
  print 'Generating likelihoods...'
  de.genLike(theta, data, params, log_dens_like, fit, nJobs)
  print 'Done!'

#cull the stuck chains: remove index of all good chains
from statsmodels.robust.scale import mad
def cullStuck(theta, val = 3.5, replace = False):
  s = mad(theta, axis = 1)
  meanSD = mad(theta, axis = (0,1)) #get the avg. SD over chains
  idx = (s < (meanSD * val)) & (s > (meanSD / val))
  idx = np.mean(idx, axis = 1) == 1.0

  if replace == False: return theta[idx]
  else:
    if np.sum(idx) > 0:
      nos = np.where(idx == True)[0]
      theta[idx == False] = theta[np.random.choice(nos, np.sum(idx == False), replace = True)]
    return theta
  


