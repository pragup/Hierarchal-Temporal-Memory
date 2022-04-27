# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 21:26:14 2021

@author: Prashant Gupta
"""
import numpy as np
import math

class spatialPooler:
    
    # Initialize all the parameters
    def __init__(self, inputSDRs_, batchSize_, nEpoch_, na_, nc_, nps_, permInc_, permDec_, proxSynThres_, windPermInitial_, proxDendriThres_, desirColAct_, maximumBoost_, dutyCyclePeriod_, minActLevel_ = 0.01, permBoostScalFact_=0.1):
        # Parameters
        self.inputSDRs = inputSDRs_
        self.ns = len(inputSDRs_) # number of input samples
        self.na = na_ # number of attributes in a input
        self.nc = nc_ # number of columns
        self.batchSize = batchSize_
        self.nBatch = int(self.ns/self.batchSize)
        self.nps = nps_ # number of proximal synapses per column 0< nps_<=na_ 
        self.permInc = permInc_ # permanence increment amount [0, 1]
        self.permDec = permDec_ # permanence decrement amount [0, 1]
        self.proxSynThres = proxSynThres_ # proximal synapse activation threshold (0, 1]
        self.windPermInitial = windPermInitial_ # window of permanence initialization
        self.proxDendriThres = proxDendriThres_ # proximal dentrite threshold or overlap threshold
        self.desirColAct = desirColAct_ # desired column activity level 0 < desirColAct_ < nc_ 
        self.minActScalFact = minActLevel_ # minimum activity level scaling factor
        self.permBoostScalFact = permBoostScalFact_ # permanence boosting scaling factor 
        self.maximumBoost = maximumBoost_ # maximum boost limit 
        self.dutyCyclePeriod = dutyCyclePeriod_ # duty cycle period for boosting
        self.inhibitRad = 1 # inhibition radius
        
        #Initialize Arrays
        self.delta = None # increment and decrement in synapsis
        self.dphi = None # change in dphi
        self.phi = None # proximal synapsis with permanance values
        self.C = None # proximal synapsis
        self.genInitialCAndPhi() # initialize C and Phi
        
        self.X = None # per batch active synapses of input for a given columns
        self.notX = None # per batch active synapses of input for a given columns
        
        self.gamma = None # for inhibition
        
        self.distColumnToSyn = np.zeros(self.C.shape)
        self.genDistColumnToSyn()
        self.b = np.zeros((1, self.nc))       
        # self.b = np.ones((self.batchSize, self.nc)) 
        
        self.H = None # neighborhood
        self.columnNeighborhood() # compute neighborhood
        
        self.Y = None
        self.alpha = None
        self.cHat = None
        
        self.nua = np.zeros((1, self.nc)) # active duty cycle for all column
        self.counter = 0 # counter for batch or each training example 
        self.nuMin = np.zeros((1, self.nc)) # minimum active duty cycle for all column
        self.nuo = np.zeros((1, self.nc)) # overlap duty cycle for all column 
        
    def poolerSolve(self, n_epoch):
        
        for epoch_ in range(n_epoch):
            for batchIndex in range(self.nBatch):
                self.calculateY()
                self.genX(batchIndex)
                self.overlap()
                self.inhibition()
                self.learning()
                print("epoch:", epoch_, "ith Batch Finished:", batchIndex)
                # if batchIndex == 0: break
        self.calculateY() # Final connectivity matrix
        
    def calculateY(self):
        self.Y = (self.phi >= self.proxSynThres).astype(int) 

    def genInitialCAndPhi(self):
        phi_ = np.zeros((self.nc, self.na))
        for i_ in range(self.nc): phi_[i_, np.random.choice(self.na, self.nps, replace = False)] = 1
        self.C = phi_
        phivalue_ = np.random.uniform(self.proxSynThres - self.windPermInitial, self.proxSynThres + self.windPermInitial, self.nc * self.na)
        phivalue_ = phivalue_.reshape((self.nc, self.na))
        self.phi = phi_ * phivalue_
    
    def genX(self, batchIndex):
        
        # active synapses of input for a given columns
        
        inputData_ = self.inputSDRs[batchIndex*self.batchSize:(batchIndex + 1)*self.batchSize, :]
        notInputData_ = (inputData_<=0).astype(int)
        self.X = np.zeros((inputData_.shape[0] * self.nc, self.na))
        self.notX = np.zeros((inputData_.shape[0] * self.nc, self.na))
        for index_ in range(inputData_.shape[0]):
           self.X[index_*self.nc : (index_+1)*self.nc] = self.C * inputData_[index_]
           self.notX[index_*self.nc : (index_+1)*self.nc] = self.C * notInputData_[index_]
           
        self.notX = (self.X<=0).astype(int)
        
    
    def genDistColumnToSyn(self):
        
        list1_ = []
        
        for i_ in range(self.na):
            list1_.append(np.where(self.C[:, i_] == 1)[0][0])
        list1_ = np.array(list1_)
        
        distM_ = np.zeros(self.C.shape)
        for i_ in range(self.nc):
            dlist1_ = list1_ - i_
            dist_ = 2 * (dlist1_!=0).astype(int)
            distM_[i_, :] = dist_
        
        self.distColumnToSyn = distM_

    def estimateParameters(self):
        pass
    
    def columnNeighborhood(self, globalInhibition = True):
        
        H_ = np.dot(self.C, self.C.T) #
        
        if not globalInhibition:
            H = np.zeros((self.nc, self.nc))
            for i_ in range(self.nc):
                rowIndexList_ = np.argsort(H_[i_, :])[self.nc - 1 - self.inhibitRad : self.nc - 1]
                H[i_, rowIndexList_] = 1
            self.H = H 
        else:
            self.H = (H_ > 0).astype(int)
    
    def overlap(self):
        # inputData_ = self.X[batchIndex*self.batchSize:(batchIndex + 1)*self.batchSize, :]
        batchSize_ = int(len(self.X)/self.nc)
        alphaHat_ = np.zeros((batchSize_, self.nc))
        count_ = 0
        for index_ in range(0, len(self.X), self.nc):
            alphaHat_[count_] = np.sum(self.X[index_: index_ + self.nc] * self.Y, axis = 1)
            count_ = count_ + 1
            
        alphaHatMask_ = (alphaHat_ >= self.proxDendriThres).astype(int)
        alpha_ = alphaHatMask_ * alphaHat_  
        alpha_ = alpha_ * self.b
        
        self.alpha = alpha_
        
    def inhibition(self):
        gamma_ = np.zeros(self.alpha.shape)
        for i_ in range(len(self.alpha)):    
            temp_ = -self.H * self.alpha[i_]
            for j_ in range(self.nc):
                temp__ = max(-np.partition(temp_[j_, :], self.desirColAct)[self.desirColAct], 1)
                gamma_[i_, j_] = temp__
        
        self.gamma = gamma_    
        self.cHat = (self.alpha >= gamma_).astype(int)
                
    def learning(self):
        phiplusX = self.permInc * self.X
        phiminusnotX = self.permDec * self.notX
        self.delta = (phiplusX - phiminusnotX)
        dphi_ = np.zeros((self.na, self.nc))
        count_ = 0
        for index_ in range(0, len(self.X), self.nc):
            dphi_ = dphi_ + (self.cHat[count_] * self.delta[index_: index_ + self.nc].T) 
            count_ = count_ + 1
            
        self.dphi = dphi_.T    
        self.phi = np.clip(self.phi + self.dphi, 0, 1)
        
        self.nuMin =  self.minActScalFact * np.max(self.H * self.nua, axis = 1).reshape(self.nua.shape)
        
        self.counter = self.counter + 1
        
        self.updateActiveDutyCycle()
        self.updateBoost()
        self.updateOverlapDutyCycle()
        # phiTemp_ = self.phi + self.permBoostScalFact * self.proxSynThres *(self.nuo.T < self.nuMin.T).astype('int') 
        # self.phi = np.clip(phiTemp_, 0, 1)
        # self.updateInhibitRad()
        
    def updateActiveDutyCycle(self):
        batchSize_ = int(len(self.X)/self.nc)
        cSumAvg_ = (1 / batchSize_) * np.sum(self.cHat, axis = 0)
        self.nua = (1/self.counter) * (cSumAvg_ + (self.counter - 1) * self.nua)
        
    def updateBoost(self):
        b_ = np.zeros(self.nua.shape)
        for i_ in range(self.nc):
            
            if self.nuMin[0, i_] == 0:    
                b_[0, i_] = self.maximumBoost 
            elif self.nua[0, i_] > self.nuMin[0, i_]:
                b_[0, i_] = 1
            else:
                b_[0, i_] = (self.nua[0, i_] / self.nuMin[0, i_])*(1 - self.maximumBoost) + self.maximumBoost         
        self.b = b_
        
    def updateOverlapDutyCycle(self):
        batchSize_ = int(len(self.X)/self.nc)
        alphaSumAvg_ = (1 / batchSize_)*np.sum(self.alpha, axis = 0)
        self.nuo = (1/self.counter) * (alphaSumAvg_ + (self.counter - 1) * self.nua)
    
    def updateInhibitRad(self):
        
        D_ = self.distColumnToSyn * self.Y
        f_ = np.round(D_.sum()/ max(1, self.Y.sum()), 0) #math.floor(D_.sum()/ max(1, self.Y.sum()))
        self.inhibitRad = max(1, f_).astype('int')
    
    def getOSDRS(self, data_, numWinners_= 10):
        
        osdrsData_ = np.zeros((data_.shape[0], self.Y.shape[0]))
        #print(osdrsData_.shape)
        dataOSDRSIndices_ = np.argpartition(np.dot(-data_, self.Y.T), numWinners_, axis = 1)[:, :numWinners_]        # osdrsData_[dataOSDRSIndices_] = 1
        
        for i_ in range(len(data_)):
            osdrsData_[i_, dataOSDRSIndices_[i_]] = 1
        
        print(dataOSDRSIndices_.shape)
        return osdrsData_
        
    