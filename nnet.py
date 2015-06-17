import numpy as np
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle
import pdb

class nnet(object):

    def __init__(self):

        # Network parameters
        self.units = 512
        self.layers = 1
        self.imSize = 64*64
        self.alpha = .001
        self.encLen = 10
        self.decLen = 10
        self.futLen = 10
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
        self.decImW = np.random.uniform(-scale2,scale2,(self.imSize,self.units))
        self.decImB = np.zeros((self.imSize,1))

        # Future
        self.futOut = []
        self.futIn = []
        self.futImIn = []
        self.futImOut = []
        self.futW = np.random.uniform(-scale2,scale2,(self.units,self.units))
        self.futImW = np.random.uniform(-scale2,scale2,(self.imSize,self.units))
        self.futImB = np.zeros((self.imSize,1))

        # Other
        self.updates = []
        self.loss = []
        
        # Load Dataset
        self.trainSetSize = 300
        self.testSetSize = 300

        if (os.path.isfile('data.p')):
            dataFile = open('data.p','rb')
            self.trainSet = pickle.load(dataFile)
            self.testSet = pickle.load(dataFile)
        else:
            self.trainSet = self.createTrainSet()
            dataFile = open('data.p','w')
            pickle.dump(self.trainSet,dataFile)
            self.testSet = self.createTestSet()
            pickle.dump(self.testSet,dataFile)

    def createTrainSet(self):
        
        carColor = .5
        backgroundColor = 1
        trainSet = []
        roads = [0] * 4
        speed = [0] * 4
        position = [0] * 4
        count = 0

        while (count < 10):
            dim = np.sqrt(self.imSize)
            image = np.ones(dim,dim) * backgroundColor
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
            trainSet.append(np.reshape(image,(self.imSize,1)))
            count += 1
        return trainSet

    def createTestSet(self):        
        
        return []

    def act(self,z):

        return 1/(1 + np.exp(-1 * z))

    def der(self,z):

        return self.act(z) * (1 - self.act(z))

    def calculateLoss(self,decImages,futImTruth):

        decTotal = 0.0
        decNorm = 0.0
        for i in range(0,len(decoderImages)):
            norm = np.linalg.norm(decImages[i] - self.decImOut[i])
            decTotal = decTotal + 1.0/2 * np.square(norm)
            decNorm = decNorm + np.sum(decImages[i])

        futTotal = 0.0
        futNorm = 0.0
        for i in range(0,len(decoderImages)):
            norm = np.linalg.norm(futImTruth[i] - self.futImOut[i])
            futTotal = futTotal + 1.0/2 * np.square(norm)
            futNorm = futNorm + np.sum(futImTruth[i])

        self.loss = [decTotal,decTotal/decNorm,futTotal,futTotal/futNorm]

    def forwardProp(self,encImTruth,decImTruth,futImTruth):

        # Encoder    
        self.encOut = np.zeros((self.units,len(encImTruth)))
        self.encInIm = np.zeros((self.units,len(encImTruth)))
        self.encInPast = np.zeros((self.units,len(encImTruth)))
        self.encIn = np.zeros((self.units,len(encImTruth)))

        for i in range(0,len(encoderImages)):
            self.encInIm[:,[i]] = np.dot(self.encImW,encImTruth[i]) + self.encImB
            self.encInPast[:,[i]] = np.dot(self.encW,self.encOut[:,[i-1]])
            self.encIn[:,[i]] = self.encInIm[:,[i]] + self.encInPast[:,[i]]
            self.encOut[:,[i]] = self.act(self.encIn[:,[i]])

        # Decoder
        self.decIn = np.zeros((self.units,len(decImTruth)))
        self.decIn[:,[0]] = np.dot(self.encDecW,self.encOut[:,[-1]]) + self.decB
        self.dec = np.zeros((self.units,len(decImTruth)))
        self.decImIn = np.zeros((self.imSize,len(decImTruth)))
        self.decImOut = np.zeros((self.imSize,len(decImTruth)))

        for i in range(0,len(decImTruth)):
            self.decOut[:,[i]] = self.act(self.decIn[:,[i]])
            weightedImage = np.dot(self.decImW,self.decOut[:,[i]])
            self.decImIn[:,[i]] =  weightedImage + self.decImB
            self.decImOut[:,[i]] = self.act(self.decImOut[:,[i]])
            if (i < len(decImTruth) - 1):
                self.decIn[:,[i+1]] = np.dot(self.decW,self.decOut[:,[i]])

        # Future
        self.futIn = np.zeros((self.units,len(futImTruth)))
        self.futIn[:,[0]] = np.dot(self.encFutW,self.encOut[:,[-1]]) + self.futB
        self.futOut = np.zeros((self.units,len(futImTruth)))
        self.futImIn = np.zeros((self.imSize,len(futImTruth)))
        self.futImOut = np.zeros((self.imSize,len(futImTruth)))

        for i in range(0,len(futImTruth)):
            self.futOut[:,[i]] = self.act(self.futIn[:,[i]])
            weightedImage = np.dot(self.futImW,self.futOut[:,[i]])
            self.futImIn[:,[i]] = weightedImage + self.futImB
            self.futImOut[:,[i]] = self.act(self.futImIn[:,[i]])
            if (i < len(futImTruth) - 1):
                self.futIn[:,[i+1]] = np.dot(self.futW,self.futOut[:,[i]])

        self.calculateLoss(decImTruth,futImTruth)

    def backProp(self,encImTruth,decImTruth,futImTruth):

        # Decoder
        delDec = np.zeros((self.units,len(decImTruth)))
        delDecIm = np.zeros((self.units, len(decImTruth)))
        diff = self.decImOut[:,[-1]] - self.decImTruth[-1]
        delDecIm[:,[-1]] = np.dot(diff.T,self.der(self.decImIn[:,[-1]]))
        delDec[:,[-1]] = np.dot(np.dot(self.decImW.T,delDecIm[:,[-1]]).T,\
                               self.der(self.decIn[:,[-1])))

        for i in range(0,len(decImTruth) - 1)[::-1]:
            diff = self.decImOut[:,[i]] - self.decImTruth[i]
            delDecIm[:,[i]] = np.dot(diff.T,self.der(self.decImIn[:,[i]]))
            delFromIm = np.dot(np.dot(self.decImW.T,delDecIm[:,[i]]).T,\
                               self.der(self.decIn[:,[i]]))
            delFromTime = np.dot(np.dot(self.decW.T,delDec[:,[i+1]]).T,\
                                 self.der(self.decIn[:,[i]]))
            delDec[:,[i]] = delFromTime + delFromIm

        # Future
        delFut = np.zeros((self.units,len(futImTruth)))
        delFutIm = np.zeros((self.units, len(futImTruth)))
        diff = self.futImOut[:,[-1]] - self.futImTruth[-1]
        delFutIm[:,[-1]] = np.dot(diff.T,self.der(self.futImIn[:,[-1]]))
        delTimesWeight = np.dot(self.futImW.T,delFutIm[:,[-1]])
        delFut[:,[-1]] = np.dot(delTimesWeight.T,self.der(self.futIn[:,[-1])))

        for i in range(0,len(futImTruth) - 1)[::-1]:
            delTimesWeight = np.dot(self.futW.T,delFut[:,[i+1]])
            delFromTime = np.dot(delTimesWeight.T, self.der(self.futIn[:,[i]]))
            diff = self.futImOut[:,[i]] - self.futImTruth[i]
            delFutIm[:,[i]] = np.dot(diff.T,self.der(self.futImIn[:,[i]]))
            delTimesWeight = np.dot(self.decImW.T,delFutIm[:,[i]])
            delFutIm = np.dot(delTimesWeight.T,self.der(self.futIn[:,[i]]))
            delFut[:,[i]] = delFromTime + delFromIm

        # Encoder
        delEnc = np.zeros((self.units, len(encImTruth)))
        delTimesWeightDec  = np.dot(self.encDetW.T,delDec[:,[0]])
        delTimesWeightFut = np.dot(self.encFutW.T,delFut[:,[0]]))
        delTimesWeight = delTimesWeightDec + delTimesWeightFut
        delEnc[:,[-1]] = np.dot(delTimesWeight.T, self.der(self.encIn[:,[-1]]))
                             
        for i in range(0,len(encImTruth) - 1)[::-1]:
            delTimesWeight = np.dot(self.encW.T,delEnc[:,[i+1]])
            delEnc[:,[i]] = np.dot(delTimesWeight.T, self.der(self.encIn[:,[i]]))

        # Encoder Update
        updateWI = np.sum(np.dot(delEnc,self.encImTruth))
        self.encImW = self.weightInput - self.alpha*updateWI

        updateBI = np.sum(delEnc)

        self.encImB = self.biasEncoder - self.alpha*updateBI

        updateWE = np.sum(np.dot(delEnc[1::],self.encOut[0:-1]))
        self.encW = self.encW - self.alpha*updateWE

        updateWencDec = np.sum(np.dot(delDec[0],self.encOut[-1]))
        self.encDecW = self.encDecW - self.alpha*updateWB

        updateWencFut = np.sum(np.dot(delFut[0],self.encOut[-1]))
        self.encFutW = self.encFutW - self.alpha*updateWB

        # Decoder Update
        updateWD = np.sum(np.dot(delDec[1::],self.dec[0:-1]))
        self.decW = self.decW - self.alpha*updateWD

        updateWDI = np.sum(np.dot(delDecIm,self.dec))
        self.decImW = self.decImW - self.alpha*updateWDI

        updateBDI = np.sum(delDecIm)
        self.decImB = self.decImB - self.alpha*updateBDI

        updateBD = delDec[0]
        self.decB = self.decB - self.alpha*updateBD                        
        
        # Future Update
        updateWF = np.sum(np.dot(delFuture[1::],self.futOut[0:-1]))
        self.futW = self.futW - self.alpha*updateWF

        updateWPI = np.sum(np.dot(delFutIm,self.futOut))
        self.futImW = self.futImW - self.alpha*updateWPI

        updateBPI = np.sum(delFutIm)
        self.futImB = self.futImB - self.alpha*updateBPI

        updateBF = delFut[0]
        self.futB = self.futB - self.alpha*updateBF

        self.updates = [updateWI,updateBI,updateWE,updateWencDet,\
                       updateWencFut,updateBD,updateBF,updateWD,updateWF]

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

    def dumpImages(self,images):

        for i in range(0,len(images)):
            plt.imshow(self.reshapeImageWithBorder(images[i]))
            plt.savefig("train_" + str(i) + ".png")

    def run(self):

        for e in range(0,self.epochs):
            iteration = 0
            start = 0
            while (start < len(self.trainSet) - self.encLen - self.futLen):
                encoderImages = self.trainSet[start,start+self.encLen]
                decoderImages = encoderImages[::-1]
                futureImages = self.trainSet[start+self.encLen,\
                               start+self.encLen+self.futLen]                 
                self.forwardProp(encoderImages, decoderImages, futureImages)
                print "Epoch: %02d, Iter: %04d" % (e, iteration)
                self.backProp(encoderImages, decoderImages, futureImages)
                start = start + self.encLen
                iteration = iteration + 1


# , Dec: %06d, DecN: %d, Fut: %06d, FutN: %d, updateWI: %d, updateBI: %d, 
# updateWE: %d, updateWencDet: %d, updateWencFut %d, updateBD: %d, updateBF: %d
# , updateWD: %d, updateWF: %d" % (e, iteration, self.loss[0], self.loss[1], 
# self.loss[2], self.loss[3], self.update[0], self.update[1], self.update[2], 
# self.update[3], self.update[4], self.update[5], self.update[6], 
# self.update[7],self.update[8])

        
