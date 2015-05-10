import tools
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import math

class nnet(object):

    def __init__(self):

        self.outputNum = 3
        self.units = 16
        self.learningRate = .5
        self.activations = []
        self.inputs = []
        self.weights = []
        self.bias = []
        self.image = np.zeros((self.units))
        self.image[0] = 1
        self.images = [self.image]
        for i in range(0,self.outputNum):
            self.weights.append(np.zeros((self.units,self.units)))
            self.bias.append(np.zeros((self.units)))
            temp = np.zeros((self.units))
            temp[i+1] = 1
            self.images.append(temp)
        
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

    def forwardProp(self):

        self.activations = [self.images[0]]
        self.inputs = []

        for W, b in zip(self.weights,self.bias):
            z = np.dot(W,self.activations[-1]) + b
            self.inputs.append(z)
            self.activations.append(self.act(z))

    def backProp(self):

        deltas = [np.dot(-(self.activations[-1] - self.images[-1]),self.der(self.inputs[-1]))]
        
        for i in range(1,len(self.weights))[::-1]:
            deltaRight = np.dot(self.weights[i].T * deltas[-1], self.der(self.inputs[i-1]))
            deltaBottom = np.dot(-(self.activations[i] - self.images[i]),self.der(self.inputs[i-1]))
            deltas.append(deltaRight + deltaBottom)
        for i in range(0,len(self.weights)):
            updateW = np.dot(deltas[i],self.activations[i].T)
            self.weights[i] = self.weights[i] - self.learningRate*updateW
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


        
