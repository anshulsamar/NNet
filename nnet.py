import tools
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import math

class nnet(object):

    def __init__(self):

        self.units = 2048
        self.layers = 1
        self.inputSize = 64*64 #64 x 64 images, 
        self.learningRate = .5
        self.seqLen = 10
        self.activations = []
        self.inputs = []
        self.encoder = []
        self.decoder = []
        self.inputDecoder = []
        self.future = []
        self.inputFuture = []
        self.scale = 1/np.sqrt(64) #intialize weights to [-1/sqrt(fan-in),1/sqrt(fan-in)]
        self.weightEncoder = np.random.uniform(-self.scale,self.scale,self.units,self.units)
        self.weightDecoder = np.random.uniform(-self.scale,self.scale,self.units,self.units)
        self.weightFuture = np.random.uniform(-self.scale,self.scale,self.units,self.units)
        self.weightBetween = np.random.uniform(-self.scale,self.scale,self.units,self.units)
        self.weightInput = np.random.uniform(-self.scale,self.scale,self.units,self.inputSize)
        self.bias = np.zeros((self.units))
        self.createDataset()

    def createDataset(self):
        
    def act(self,z):

        return np.divide(1,(np.add(1,np.exp(np.multiply(-1,z)))))

    def der(self,z):

        return self.act(z) * (1 - self.act(z))

    def cost(self):

        c = []
        for i in range(0,self.outputNum+1):
            dif = self.activations[i] - self.images[i]
            c.append(1.0/2 * np.dot(dif,dif))
        return c

    # factor in bias, reintialize matrixes back to 0

    def forwardProp(self,images,imagesDecoder,imagesFuture):

        # Encoder
        
        self.encoder = [np.zeros((self.units,self.units))]

        for i in range(0,len(images)):
            self.inputImage.append(np.dot(self.weightInput,images[i]))
            self.inputPast.append(np.dot(self.weightEncoder,self.encoder[-1]))
            self.inputEncoder.append(self.inputPast[-1] + self.inputPast[-1])
            self.encoder.append(self.act(self.inputEncoder[-1]))

        self.encoder.remove(0)

        # Decoder

        self.inputDecoder.append(np.dot(self.weightBetween,self.encoder[-1])])
        self.decoder = []

        for i in range(0,len(imagesDecoder)):
            self.decoder.append(self.act(self.inputDecoder[-1]))
            self.inputDecoder.append(np.dot(self.weightDecoder,self.decoder[-1]))

    # Function: backProp(self, images, imagesDecoder, imagesFuture):
    # ------------------------------------------------------
    # BPTT using 'imagesDecoder' as groundtruth for decoder and 
    # 'imagesFuture' as groundtruth for future. Images are
    # the input images to encoder. Euclidean loss function.

    def backProp(self,imagesEncoder,imagesDecoder,imagesFuture):

        # Decoder

        deltasDecoder.append(np.dot((self.decoder[-1] - self.imagesDecoder[-1]),self.der(self.inputDecoder[-1]))])

        for i in range(0,len(imagesDecoder) - 1)[::-1]:
            deltaTimeDecoder = np.dot(self.weightDecoder.T * deltasDecoder[-1], self.der(self.inputDecoder[i]))
            deltaImageDecoder = np.dot((self.decoder[i] - self.images[i]),self.der(self.inputDecoder[i]))
            deltasDecoder.append(deltaImageDecoder + deltaTimeDecoder)

        # Encoder

        delta = np.dot(self.weightBetween.T * deltaDecoder[-1], self.der(self.inputEncoder[-1]))
        deltaTimeEncoder = [np.dot(delta,self.der(self.inputTime[-1]))]
        deltaImageEncoder = [np.dot(delta,self.der(self.inputImage[-1]))]
                             
        for i in range(0,len(imagesEncoder) - 1)[::-1]:
            delta = np.dot(self.weightEncoder.T * deltaTime[-1], self.der(self.inputEncoder[i])))
            deltaTimeEncoder.append(np.dot(delta,self.der(self.inputTime[i])))
            deltaImageEncoder.append(np.dot(delta,self.der(self.inputImage[i])))

        updateW = np.sum(np.dot(deltasDecoder[0:-1],self.decoder[-1::])
        self.weightDecoder = self.weightDecoder - self.learningRate*updateW

        updateW = np.sum(np.dot(deltasDecoder[-1],self.encoder[-1])
        self.weightBetween = self.weightBetween - self.learningRate*updateW

        updateW = np.sum(np.dot(deltasTimeEncoder[0:-1],self.encoder[-1::])
        self.weightEncoder = self.weightEncoder - self.learningRate*updateW

        updateW = np.sum(np.dot(deltasImageEncoder[0:-1],self.imagesEncoder[-1::])
        self.weightInput = self.weightInput - self.learningRate*updateW

        updateB = deltas[i]
        self.bias[i] = self.bias[i] - self.learningRate*updateB

    def viewImage(self,i):
        array = np.reshape(self.images[i],(np.sqrt(self.units),np.sqrt(self.units)))
        plt.imshow(array,cmap=plt.cm.gray)
        plt.show()

    def viewOutput(self,i):
        array = np.reshape(self.activations[i],(np.sqrt(self.units),np.sqrt(self.units)))
        plt.imshow(array,cmap=plt.cm.gray)
        plt.show()

    def run(self):

        while(True):
            self.forwardProp()
            self.backProp()
            print self.cost()


        
