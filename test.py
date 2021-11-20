# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 16:39:36 2021
@author: asus
"""
#%%
import spatialPooler as sp
import numpy as np
import idx2numpy
#%%
####################
#### MNIST DATA ####
####################
pixelThr_ = 128
train_data = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
test_data = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
train_label = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')
test_label =  idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')
trainISDRS = (train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])>=pixelThr_).astype(int)
testISDRS = (test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])>=pixelThr_).astype(int)

#%%
# density_ = 0.3
# np.random.seed(0)
# n_examples = 20
# width = 100
# inputSDRs_ = np.random.choice([1, 0], n_examples * width, p =[density_, 1 - density_]).reshape(n_examples, width)
# n_examples = 2
# testExamples = np.random.choice([1, 0], n_examples * width, p =[density_, 1 - density_]).reshape(n_examples, width)
#%%
# Sampled data #
np.random.seed(0)

trainISDRSSamples = trainISDRS[:6000]
trainLabelsSamples = train_label[:6000]
testISDRSSamples = testISDRS
testLabelsSamples = test_label

nc_ = 512 * 4
na_ = trainISDRSSamples.shape[1] #trainISDRS.shape[1]
nps_ = 40
proxDendriThres_ = 10
proxSynThres_ = 0.5
windPermInitial_ = 0.05
desirColAct_ = 182 #int(0.02 * nc_)
permInc_ = 0.03
permDec_ = 0.05
maximumBoost_ = 10
dutyCyclePeriod_ = 100
# inhibitRadius_ = 2
batch_ratio = 0.01
batchSize_ = int(batch_ratio * len(trainISDRSSamples))
testWinnersRatio_ = 0.10 
numWinners_ = testWinnersRatio_ * nc_
nEpoch_ = 30
sp01 = sp.spatialPooler(trainISDRSSamples, batchSize_, nEpoch_, na_, nc_, nps_, permInc_, permDec_, proxSynThres_, windPermInitial_, proxDendriThres_, desirColAct_, maximumBoost_, dutyCyclePeriod_)
sp01.poolerSolve(nEpoch_)
#%%
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
train_osdrs = sp01.getOSDRS(trainISDRSSamples, numWinners_= 204)
test_osdrs = sp01.getOSDRS(testISDRSSamples, numWinners_= 204)
train_osdrs_new = np.concatenate((trainISDRSSamples, train_osdrs), axis = 1)
test_osdrs_new = np.concatenate((testISDRSSamples, test_osdrs), axis = 1)
logReg = LogisticRegression(random_state=0, max_iter=300, C = 1)
logReg.fit(train_osdrs_new, trainLabelsSamples)
pred_labels = logReg.predict(test_osdrs_new)
acc = accuracy_score(testLabelsSamples, pred_labels)

#%%
import cProfile
cProfile.run('sp01.poolerSolve(nEpoch_)')
# print(np.count_nonzero((sp01.phi[0, :]>0)))
