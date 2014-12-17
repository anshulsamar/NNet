import tools
import numpy as np
import cudamat as cm

def class NNet:

    def __init__(self, params):

        self.params = params
        self.weights = []
        self.bias = []
        for pair in zip(layers[0:-1],layers[1:]):
            #create matrics and bias terms
        for W in weights:
            initializeWeights(W)
        for b in bias:
            initializeBias(b)
    
    def initializeWeights(W):

    def initializeBias(b):
        
    def getImage(im):

    def act(z):

    def der(z):

    def updateWeight(W, update):

    def updateBias(b, update):
        
    def calculateCost(image, output):

    def prop(image):

        activations = [image]
        inputs = []

        for W, b in zip(self.weights,self.bias):
            z = W*activations[-1] + b
            inputs.append(z)
            activations.append(act(z))

        output = activations.remove(-1)
        cost = calculateCost(image,output)
        deltas = [-(output - image)*der(inputs[-1])]
        
        for i in range(1,len(weights)).reverse():
            deltas.insert(weights[i].T * deltas[0] .* der(i-1),0)
        for i in range(0,len(weights)):
            updateW = deltas[i]*a[i].T
            updateWeight(weights[i],updateW)
            updateB = deltas[i]
            updateBias(bias[i],updateB)

        return cost

    def run():
        for epoch in self.epochs:
            for im in range(0,self.numImages):
                image = getImage(im)
                cost = prop(image)
                print "Iteration " + im + " Cost: " + cost
        
        
