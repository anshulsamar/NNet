import numpy as np
import os
import pickle
import pdb

# change number of units
# change training set size
# change enc len dec len and the debug train set splicing

class nnet(object):

    def __init__(self):

        # Network parameters
        self.units = 512
        self.layers = 1
        self.imSize = 64*64
        self.alpha = 1e-3
        self.encLen = 1
        self.decLen = 1
        self.futLen = 1
        self.trainLen = 100
        self.epochs = 1
        scale1 = 1/np.sqrt(self.imSize)
        scale2 = 1/np.sqrt(self.units)

        # Encoder
        self.encOut = []
        self.encIn = [] 
        self.encInIm = []
        self.encInPast = [] 
        self.encImW = np.random.uniform(-scale1,scale1,(self.units,self.imSize))
        self.encW = np.random.uniform(-scale2,scale2,(self.units,self.units))
        self.encImB = np.zeros((self.units,1))

        # From Encoder -> Decoder/Future
        self.encDecW = np.random.uniform(-scale2,scale2,(self.units,self.units))
        self.encFutW = np.random.uniform(-scale2,scale2,(self.units,self.units))
        self.decB = np.zeros((self.units,1))
        self.futB = np.zeros((self.units,1))

        # Decoder
        self.decOut = []
        self.decIn = []
        self.decImIn = []
        self.decImOut = []
        self.decW = np.random.uniform(-scale2,scale2,(self.units,self.units))

        # Future
        self.futOut = []
        self.futIn = []
        self.futImIn = []
        self.futImOut = []
        self.futW = np.random.uniform(-scale2,scale2,(self.units,self.units))

        # Output image
        self.outImW = np.random.uniform(-scale2,scale2,(self.imSize,self.units))
        self.outImB = np.zeros((self.imSize,1))

        # Update and Loss
        self.updateEncImW = None
        self.updateEncImB = None
        self.updateEncW = None
        self.updateEncDecW = None
        self.updateEncFutW = None
        self.updateDecW = None
        self.updateDecB = None
        self.updateFutW = None
        self.updateFutB = None
        self.updateOutImW = None
        self.updateOutImB = None
        self.loss = []
        
        # Output and Data Files
        self.decOutFile = 'decode/decode.p'
        self.futOutFile = 'future/future.p'
        self.decFileHand = ''
        self.futFileHand = ''

        if (os.path.isfile(self.decOutFile)):
            os.remove(self.decOutfile)
        if (os.path.isfile(self.futOutFile)):
            os.remove(self.futOutFile)

        self.dataFile = 'data/data.p'
        self.numDataFiles = 5
        self.imPerFile = 100

        #Other
        self.epsilon = 1e-5

    def act(self,z):

        return 1/(1 + np.exp(-1 * z))

    def der(self,z):

        return self.act(z) * (1 - self.act(z))

    def cost(self,imTruth,imOut):

        cost = 0

        for i in range(0,np.shape(imTruth)[1]):
            norm = np.linalg.norm(imTruth[:,[i]] - imOut[:,[i]])
            cost = cost + 1.0/2 * np.square(norm)

        return cost

    def forwardProp(self,encImTruth,decImTruth,futImTruth,fileCount,write=True):

        # Encoder    
        self.encOut = np.zeros((self.units,self.encLen))
        self.encInIm = np.zeros((self.units,self.encLen))
        self.encInPast = np.zeros((self.units,self.encLen))
        self.encIn = np.zeros((self.units,self.encLen))


        for i in range(0,self.encLen):
            self.encInIm[:,[i]] = np.dot(self.encImW,encImTruth[:,[i]]) \
                                  + self.encImB
            self.encInPast[:,[i]] = np.dot(self.encW,self.encOut[:,[i-1]])
            self.encIn[:,[i]] = self.encInIm[:,[i]] + self.encInPast[:,[i]]
            self.encOut[:,[i]] = self.act(self.encIn[:,[i]])

        # Decoder
        self.decIn = np.zeros((self.units,self.decLen))
        self.decIn[:,[0]] = np.dot(self.encDecW,self.encOut[:,[-1]]) + self.decB
        self.decOut = np.zeros((self.units,self.decLen))
        self.decImIn = np.zeros((self.imSize,self.decLen))
        self.decImOut = np.zeros((self.imSize,self.decLen))

        for i in range(0,self.decLen):
            self.decOut[:,[i]] = self.act(self.decIn[:,[i]])
            weightedImage = np.dot(self.outImW,self.decOut[:,[i]])
            self.decImIn[:,[i]] =  weightedImage + self.outImB
            self.decImOut[:,[i]] = self.act(self.decImIn[:,[i]])
            if (i < self.decLen - 1):
                self.decIn[:,[i+1]] = np.dot(self.decW,self.decOut[:,[i]])

        # Future
        self.futIn = np.zeros((self.units,self.futLen))
        self.futIn[:,[0]] = np.dot(self.encFutW,self.encOut[:,[-1]]) + self.futB
        self.futOut = np.zeros((self.units,self.futLen))
        self.futImIn = np.zeros((self.imSize,self.futLen))
        self.futImOut = np.zeros((self.imSize,self.futLen))

        for i in range(0,self.futLen):
            self.futOut[:,[i]] = self.act(self.futIn[:,[i]])
            weightedImage = np.dot(self.outImW,self.futOut[:,[i]])
            self.futImIn[:,[i]] = weightedImage + self.outImB
            self.futImOut[:,[i]] = self.act(self.futImIn[:,[i]])
            if (i < self.futLen - 1):
                self.futIn[:,[i+1]] = np.dot(self.futW,self.futOut[:,[i]])

        if (write):
            pickle.dump(self.decImOut,self.decFileHand)
            pickle.dump(self.futImOut,self.futFileHand)
      
    def backProp(self,encImTruth,decImTruth,futImTruth):

        # Decoder
        delDec = np.zeros((self.units,self.decLen))
        delDecIm = np.zeros((self.imSize, self.decLen))
        diff = self.decImOut[:,[-1]] - decImTruth[:,[-1]]
        delDecIm[:,[-1]] = diff * self.der(self.decImIn[:,[-1]])
        delDec[:,[-1]] = np.dot(self.outImW.T,delDecIm[:,[-1]]) * self.der(self.decIn[:,[-1]])

        for i in range(0,self.decLen - 1)[::-1]:
            diff = self.decImOut[:,[i]] - decImTruth[:,[i]]
            delDecIm[:,[i]] = diff * self.der(self.decImIn[:,[i]])
            delFromIm = np.dot(self.outImW.T,delDecIm[:,[i]]) * self.der(self.decIn[:,[i]])
            delFromTime = np.dot(self.decW.T,delDec[:,[i+1]]) * self.der(self.decIn[:,[i]])
            delDec[:,[i]] = delFromTime + delFromIm

        # Future
        delFut = np.zeros((self.units,self.futLen))
        delFutIm = np.zeros((self.imSize, self.futLen))
        diff = self.futImOut[:,[-1]] - futImTruth[:,[-1]]
        delFutIm[:,[-1]] = diff * self.der(self.futImIn[:,[-1]])
        delFut[:,[-1]] = np.dot(self.outImW.T,delFutIm[:,[-1]]) * self.der(self.futIn[:,[-1]])


        for i in range(0,self.futLen - 1)[::-1]:
            diff = self.futImOut[:,[i]] - futImTruth[:,[i]]
            delFutIm[:,[i]] = diff * self.der(self.decImIn[:,[i]])
            delFromIm = np.dot(self.outImW.T,delFutIm[:,[i]]) * self.der(self.futIn[:,[i]])
            delFromTime = np.dot(self.futW.T,delFut[:,[i+1]]) * self.der(self.futIn[:,[i]])
            delFut[:,[i]] = delFromTime + delFromIm

        # Encoder
        delEnc = np.zeros((self.units, self.encLen))
        delEnc[:,[-1]] = (np.dot(self.encDecW.T,delDec[:,[0]]) + \
                              np.dot(self.encFutW.T,delFut[:,[0]])) * \
                              self.der(self.encIn[:,[-1]])
                             
        for i in range(0,self.encLen - 1)[::-1]:
            delEnc[:,[i]] = np.dot(self.encW.T,delEnc[:,[i+1]]) * self.der(self.encIn[:,[i]])
    
        # Encoder Update
        self.updateEncImW = np.dot(delEnc,encImTruth.T)
        self.encImW = self.encImW - self.alpha*self.updateEncImW

        self.updateEncImB = np.reshape(np.sum(delEnc,1),(self.units,1))
        self.encImB = self.encImB - self.alpha*self.updateEncImB

        self.updateEncW = np.dot(delEnc[:,1::],self.encOut[:,0:-1].T)
        self.encW = self.encW - self.alpha*self.updateEncW

        self.updateEncDecW = np.dot(delDec[:,[0]],self.encOut[:,[-1]].T)
        self.encDecW = self.encDecW - self.alpha*self.updateEncDecW

        self.updateEncFutW = np.dot(delFut[:,[0]],self.encOut[:,[-1]].T)
        self.encFutW = self.encFutW - self.alpha*self.updateEncFutW

        # Decoder Update
        self.updateDecW = np.dot(delDec[:,1::],self.decOut[:,0:-1].T)
        self.decW = self.decW - self.alpha*self.updateDecW

        self.updateDecB = delDec[:,[0]]
        self.decB = self.decB - self.alpha*self.updateDecB
        
        # Future Update
        self.updateFutW = np.dot(delFut[:,1::],self.futOut[:,0:-1].T)
        self.futW = self.futW - self.alpha*self.updateFutW

        self.updateFutB = delFut[:,[0]]
        self.futB = self.futB - self.alpha*self.updateFutB

        # Image Out Update
        self.updateOutImW = np.dot(delDecIm,self.decOut.T) \
                     + np.dot(delFutIm,self.futOut.T)
        self.outImW = self.outImW - self.alpha*self.updateOutImW

        self.updateOutImB = np.reshape(np.sum(delDecIm,1) + np.sum(delFutIm,1),\
                                  (self.imSize,1))
        self.outImB = self.outImB - self.alpha*self.updateOutImB

    def gradCheck(self):

        start = 0
        iteration = 0
        self.trainSet = self.loadTrainingSet(0)
        encImTruth = self.trainSet[:,start:start+self.encLen]
        decImTruth = encImTruth[:,::-1]
        futImTruth = self.trainSet[:,\
            start+self.encLen:start+self.encLen+self.futLen]

        randRow = np.random.randint(0,self.units)
        randCol = np.random.randint(0,self.imSize)
        epsilonMatrix = np.zeros((self.units,self.imSize))
        epsilonMatrix[randRow,randCol] = self.epsilon
        
        savedW = self.encImW
        self.encImW = self.encImW + epsilonMatrix
        self.forwardProp(encImTruth,decImTruth,futImTruth,0,write=False)
        costPlus = self.cost(decImTruth,self.decImOut) + self.cost(futImTruth,self.futImOut)
        
        self.encImW = savedW
        self.encImW = self.encImW - epsilonMatrix
        self.forwardProp(encImTruth,decImTruth,futImTruth,0,write=False)
        costMinus = self.cost(decImTruth,self.decImOut) + self.cost(futImTruth,self.futImOut)
        grad = (costPlus - costMinus)/(2*self.epsilon)
        self.encImW = savedW

        self.forwardProp(encImTruth, decImTruth, futImTruth,0,write=False)
        self.backProp(encImTruth, decImTruth, futImTruth)
        cost = self.cost(decImTruth,self.decImOut) + self.cost(futImTruth,self.futImOut)
        diff = np.absolute(grad - self.updateEncImW[randRow,randCol])
        weightedDiff = diff/(np.absolute(grad) + np.absolute(self.updateEncImW[randRow,randCol]))
        print "Numerical: %2.6f Backprop: %2.6f Diff: %2.8f WeightedDiff: %2.8f" % (grad,self.updateEncImW[randRow,randCol],diff,weightedDiff)
        start = start + self.encLen
        iteration = iteration + 1

    def run(self):

        for f in range(0,self.numDataFiles):

            iteration = 0
            start = 0
            self.trainSet = self.loadTrainingSet(f)
            self.decFileHand = open(self.decOutFile[:-2] + str(f) + '.p','w')
            self.futFileHand = open(self.futOutFile[:-2] + str(f) + '.p','w')

            while (start <= self.imPerFile - self.encLen):

                encImTruth = self.trainSet[:,start:start+self.encLen]
                decImTruth = encImTruth[:,::-1]
                futImTruth = self.trainSet[:,start+self.encLen:start+self.encLen+self.futLen]
                self.forwardProp(encImTruth, decImTruth, futImTruth,f,write=True)
                print "File: %02d, Iter: %04d, Dec: %2.2f, Fut: %2.2f" % \
                    (f, iteration, self.cost(decImTruth,self.decImOut),\
                     self.cost(futImTruth,self.futImOut))
                self.backProp(encImTruth, decImTruth, futImTruth)
                start = start + self.encLen
                iteration = iteration + 1

            self.decFileHand.close()
            self.futFileHand.close()
