import numpy as np
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
        self.encDecW = np.random.uniform(scale2,scale2,(self.units,self.units))
        self.encFutW = np.random.uniform(scale2,scale2,(self.units,self.units))
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
        self.updates = []
        self.loss = []
        
        # Output and Data Files
        self.decOutFile = 'decode.p'
        self.futOutFile = 'future.p'
        
        if (os.path.isfile(self.decOutFile)):
            os.remove(self.decOutfile)
        if (os.path.isfile(self.futOutFile)):
            os.remove(self.futOutFile)

        self.dataFile = 'data.p'
        self.numDataFiles = 10
        self.imPerFile = 100

        # Count
        self.epoch = 0
        self.iteration = 0

    def createTrainSet(self):
        
        carColor = .5
        backgroundColor = 1
        roads = [0] * 4
        speed = [0] * 4
        position = [0] * 4
        count = 0
        fileCount = 0

        while (fileCount < self.numDataFiles):

            trainSet = np.zeros((self.imSize,0))
            trainSetNext = np.zeros((self.imSize,0))
            count = 0

            while (count < self.imPerFile + self.futLen):
                dim = np.sqrt(self.imSize)
                image = np.ones((dim,dim)) * backgroundColor
                for i in range(0,4):
                    if roads[i] == 1:
                        if position[i] < np.sqrt(self.imSize) - 4:
                            row = i*16 + 6
                            position[i] = position[i] + speed[i]
                            image[row:row+4,position[i]:position[i]+4] = carColor

                        else:
                            roads[i] = 0
                            speed[i] = 0
                            position[i] = 0
                    else:
                        if (np.random.uniform(0,1,1) < 0.25):
                            roads[i] = 1
                            position[i] = 0
                            speedRand = np.random.uniform(0,1,1)
                            if (speedRand < 1.0/3):
                                speed[i] = 1
                            elif (speedRand < 2.0/3):
                                speed[i] = 2
                            else:
                                speed[i] = 4
                            row = i*16 + 6
                            column = 0
                            image[row:row+4,column:column+4] = carColor
                trainSet = np.hstack((trainSet,np.reshape(image,(self.imSize,1))))
                if (count >= self.imPerFile - self.futLen):
                    trainSetNext = np.hstack((trainSet,np.reshape(image,(self.imSize,1))))
                count += 1
            dFile = open(self.dataFile[:-2] + str(fileCount) + '.p','w')
            pickle.dump(trainSet,decFile)
            dFile.close()
            dFile = open(self.dataFile[:-2] + str(fileCount+1) + '.p','w')
            pickle.dump(trainSetNext,decFile)
            decFile.close()
            fileCount += 1

    def createTestSet(self):        
        
        return []

    def act(self,z):

        return 1/(1 + np.exp(-1 * z))

    def der(self,z):

        return self.act(z) * (1 - self.act(z))

    def calculateLoss(self,decImTruth,futImTruth):
        
        pdb.set_trace()

        decTotal = 0.0
        decNorm = 0.0
        for i in range(0,self.decLen):
            norm = np.linalg.norm(decImTruth[:,[i]] - self.decImOut[:,[i]])
            decTotal = decTotal + 1.0/2 * np.square(norm)
            decNorm = decNorm + np.sum(decImTruth[:,[i]])

        futTotal = 0.0
        futNorm = 0.0
        for i in range(0,self.futLen):
            norm = np.linalg.norm(futImTruth[:,[i]] - self.futImOut[:,[i]])
            futTotal = futTotal + 1.0/2 * np.square(norm)
            futNorm = futNorm + np.sum(futImTruth[:,[i]])

        self.loss = [decTotal,decTotal/decNorm,futTotal,futTotal/futNorm]

    def forwardProp(self,encImTruth,decImTruth,futImTruth,fileCount):

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
            self.decImOut[:,[i]] = self.act(self.decImOut[:,[i]])
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

        self.calculateLoss(decImTruth,futImTruth)

        decFile = open(self.decoderOutputFile[:-2] + str(fileCount) + '.p','w')
        pickle.dump(self.decOut,decFile)
        decFile.close()

        futFile = open(self.futureOutputFile[:-2] + str(filecount) + '.p','w')
        pickle.dump(self.futOut,futFile)
        futFile.close()
      

    def backProp(self,encImTruth,decImTruth,futImTruth):

        # Decoder
        delDec = np.zeros((self.units,self.decLen))
        delDecIm = np.zeros((self.imSize, self.decLen))
        diff = self.decImOut[:,[-1]] - decImTruth[:,[-1]]
        delDecIm[:,[-1]] = np.dot(diff.T,self.der(self.decImIn[:,[-1]]))
        delDec[:,[-1]] = np.dot(np.dot(self.outImW.T,delDecIm[:,[-1]]).T,\
                               self.der(self.decIn[:,[-1]]))

        for i in range(0,self.decLen - 1)[::-1]:
            diff = self.decImOut[:,[i]] - decImTruth[:,[i]]
            delDecIm[:,[i]] = np.dot(diff.T,self.der(self.decImIn[:,[i]]))
            delFromIm = np.dot(np.dot(self.outImW.T,delDecIm[:,[i]]).T,\
                               self.der(self.decIn[:,[i]]))
            delFromTime = np.dot(np.dot(self.decW.T,delDec[:,[i+1]]).T,\
                                 self.der(self.decIn[:,[i]]))
            delDec[:,[i]] = delFromTime + delFromIm

        # Future
        delFut = np.zeros((self.units,self.futLen))
        delFutIm = np.zeros((self.imSize, self.futLen))
        diff = self.futImOut[:,[-1]] - futImTruth[:,[-1]]
        delFutIm[:,[-1]] = np.dot(diff.T,self.der(self.futImIn[:,[-1]]))
        delFut[:,[-1]] = np.dot(np.dot(self.outImW.T,delFutIm[:,[-1]]).T,\
                                self.der(self.futIn[:,[-1]]))


        for i in range(0,self.futLen - 1)[::-1]:
            diff = self.futImOut[:,[i]] - futImTruth[:,[i]]
            delFutIm[:,[i]] = np.dot(diff.T,self.der(self.decImIn[:,[i]]))
            delFromIm = np.dot(np.dot(self.outImW.T,delFutIm[:,[i]]).T,\
                               self.der(self.futIn[:,[i]]))
            delFromTime = np.dot(np.dot(self.futW.T,delFut[:,[i+1]]).T,\
                                 self.der(self.futIn[:,[i]]))
            delFut[:,[i]] = delFromTime + delFromIm

        # Encoder
        delEnc = np.zeros((self.units, self.encLen))
        delEnc[:,[-1]] = np.dot((np.dot(self.encDecW.T,delDec[:,[0]]) + \
                                np.dot(self.encFutW.T,delFut[:,[0]])).T,\
                                self.der(self.encIn[:,[-1]]))
                             
        for i in range(0,self.encLen - 1)[::-1]:
            delEnc[:,[i]] = np.dot(np.dot(self.encW.T,delEnc[:,[i+1]]).T,\
                                   self.der(self.encIn[:,[i]]))
        
        # Encoder Update
        updateEncImW = np.dot(delEnc,encImTruth.T)
        self.encImW = self.encImW - self.alpha*updateEncImW

        updateEncImB = np.reshape(np.sum(delEnc,1),(self.units,1))
        self.encImB = self.encImB - self.alpha*updateEncImB

        updateEncW = np.dot(delEnc[:,1::],self.encOut[:,0:-1].T)
        self.encW = self.encW - self.alpha*updateEncW

        updateEncDecW = np.dot(delDec[:,[0]],self.encOut[:,[-1]].T)
        self.encDecW = self.encDecW - self.alpha*updateEncDecW

        updateEncFutW = np.dot(delFut[:,[0]],self.encOut[:,[-1]].T)
        self.encFutW = self.encFutW - self.alpha*updateEncFutW

        self.encoderUpdates = [updateEncImW,updateEncImB, updateEncW, \
                               updateEncDecW, updateEncFutW]
        
        # Decoder Update
        updateDecW = np.dot(delDec[:,1::],self.decOut[:,0:-1].T)
        self.decW = self.decW - self.alpha*updateDecW

        updateDecB = delDec[:,[0]]
        self.decB = self.decB - self.alpha*updateDecB

        self.decoderUpdates = [updateDecW, updateDecB]
        
        # Future Update
        updateFutW = np.dot(delFut[:,1::],self.futOut[:,0:-1].T)
        self.futW = self.futW - self.alpha*updateFutW

        updateFutB = delFut[:,[0]]
        self.futB = self.futB - self.alpha*updateFutB

        self.futureUpdates = [updateFutW, updateFutB]

        # Image Out Update
        updateOutImW = np.dot(delDecIm,self.decOut.T) \
                     + np.dot(delFutIm,self.futOut.T)
        self.outImW = self.outImW - self.alpha*updateOutImW

        updateOutImB = np.reshape(np.sum(delDecIm,1) + np.sum(delFutIm,1),\
                                  (self.imSize,1))
        self.outImB = self.outImB - self.alpha*updateOutImB

        self.outImageUpdates = [updateOutImW, updateOutImB]

        self.updates = [self.encoderUpdates, self.decoderUpdates,\
                        self.futureUpdates, self.outImageUpdates]

    def reshapeImageWithBorder(self,image):

        dim = np.sqrt(len(image))
        array = np.reshape(image,(dim,dim))
        arrayBorder = np.zeros((dim+2,dim+2))
        arrayBorder[1:1+dim,1:1+dim] = array
        return arrayBorder

    def viewImage(self,images,i):
        plt.imshow(self.reshapeImageWithBorder(images[i]),cmap = 'gray')
        plt.show()

    def viewVideo(self,images,maxFrame):
        
        fig = plt.figure()
        video = []
        for im in images:
            video.append(self.reshapeImageWithBorder(im))

        vid = ani.FuncAnimation(fig,lambda i: plt.imshow(video[i],\
              cmap='gray'),frames=maxFrame,interval=50,repeat=False)
        plt.show()


    def loadTrainingSet(self,fileNum):

        trainSet = np.zeros((self.imSize,0))
        trainFile = open(self.dataFile[:-2] + str(fileNum) + '.p','rb')
        
        while 1:
            try:
                np.hstack((self.trainSet,pickle.load(trainFile)))
            except (EOFError, UnpicklingError):
                break
                
        trainFile.close()
        return trainSet


    def loadOutput(self,fileNum):

        decIm = np.zeros((self.imSize,0))
        decFile = open(self.decoderOutputFile[:-2] + str(fileNum) + '.p','rb')

        while 1:
            try:
                np.hstack((decIm,pickle.load(decoderFile)))
            except (EOFError, UnpicklingError):
                break

        decFile.close()

        futIm = np.zeros((self.imSize,0))
        futFile = open(self.futureOutputFile[:-2] + str(fileNum) + '.p','rb')

        while 1:
            try:
                np.hstack((futIm,pickle.load(futureFile)))
            except (EOFError, UnpicklingError):
                break

        futFile.close()

        return [decIm,futIm]

    def viewOutput(self,fileNum):
        
        [decIm,futIm] = loadOutput(fileNum)
        trainSet = loadTrainingSet(fileNum)

        fig = plt.figure()
        video = []
        for i in range(0,self.imPerFile):
            dim = np.sqrt(self.imSize)
            enc = np.reshape(trainSet[:,[i]],(dim,dim))
            dec = np.reshape(trainSet[:,[self.decLen-i-1]],(dim,dim))
            futTruth = np.reshape(trainSet[:,[i+self.encLen]],(dim,dim))
            fut = np.reshape(futIm[:,[i]],(dim,dim))
            array = np.zeros((2*dim+4,2*dim+4))
            array[1:1+dim,1:1+dim] = enc
            array[1:1:dim,2+dim:2+2*dim] = futTruth
            array[2+dim:2+2*dim,1:1+dim] = dec
            array[2+dim:2+2*dim,2+dim:2+2*dim] = fut
            video.append(arrayBorder)

        vid = ani.FuncAnimation(fig,lambda i: plt.imshow(video[i],\
              cmap='gray'),frames=maxFrame,interval=50,repeat=False)
        plt.show()

    def dumpImages(self,images):

        for i in range(0,len(images)):
            plt.imshow(self.reshapeImageWithBorder(images[i]))
            plt.savefig("train_" + str(i) + ".png")

    def run(self):
        self.dataFile = 'data.p'
        self.numDataFiles = 10
        self.imPerFile = 100

        for f in range(0,self.numDataFiles):
            iteration = 0
            start = 0
            self.trainSet = self.loadTrainingSet(f)
            while (start <= self.imPerFile - self.encLen):

                encImTruth = self.trainSet[:,start:start+self.encLen]
                decImTruth = encImTruth[:,::-1]
                futImTruth = self.trainSet[:,\
                               start+self.encLen:start+self.encLen+self.futLen]
                self.forwardProp(encImTruth, decImTruth, futImTruth,fileCount)
                print "File: %02d, Iter: %04d, Dec: %7.2f, DecN: %1.2f, Fut: %7.2f, FutN: %1.2f"% (f, iteration, self.loss[0], self.loss[1], self.loss[2], self.loss[3])
                self.backProp(encImTruth, decImTruth, futImTruth)
                start = start + self.encLen
                iteration = iteration + 1
