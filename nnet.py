import numpy as np
from scipy import misc
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pickle
import pdb

# nnet
# ------------------------
# This neural network architecture is built upon "Unsupervised Learning of
# Video Representations using LSTMs" by Srivastava, et al. Let 'N' represent a
# neuron, 'I' an image, '^','->','->>','>>>','-->' all represent different
# weights and connections ('|' is not a weight). The following is a visual
# representation of this network for a sequence of three images (without bias
# terms). The output of each decoder/future neuron is compared against
# groundtruth. 
# 
# Encoder:        Decoder:
#
# N -> N -> N -> N --> N ->> N ->> N ->> N
# ^    ^    ^    ^     |     |     |     |
# I1   I2   I3   I4    I4    I3    I2    I1
#
#                 Future:

#                   --> N >>> N >>> N >>> N
#                       |     |     |     |
#                       I5    I6    I7    I8
#
# Check the README for current work list. Built on python, numpy, & cudamat.

class nnet(object):

    def __init__(self):

        # Network parameters
        self.units = 2048
        self.layers = 1
        self.inputSize = 64*64 # Must be a square if using createTrainSet()
        self.learningRate = .5
        self.encoderLen = 10
        self.decoderLen = 10
        self.futureLen = 10
        self.scale1 = 1/np.sqrt(self.inputSize)
        self.scale2 = 1/np.sqrt(self.units)
        
        # Network weights and structures. 
        # Intialized to scale 1/sqrt(fan-in)
        # Bias terms intialized to zero.
        # Weight symbols in visual rep: ^, ->, -->, ->>, >>>
        self.inputs = []
        self.encoder = []
        self.inputImage = []
        self.inputEncoderPast = []
        self.decoder = []
        self.inputDecoder = []
        self.future = []
        self.inputFuture = []
        #self.weightImage = np.random.uniform(-self.scale1,self.scale1,self.units,self.inputSize)
        #self.weightEncoder = np.random.uniform(-self.scale2,self.scale2,self.units,self.units)
        #self.weightBetween = np.random.uniform(-self.scale2,self.scale2,self.units,self.units)
        #self.weightDecoder = np.random.uniform(-self.scale2,self.scale2,self.units,self.units)
        #self.weightFuture = np.random.uniform(-self.scale2,self.scale2,self.units,self.units)
        #self.biasImage = np.zeros((self.units))
        #self.biasDecoder = np.zeros((self.units))
        #self.biasFuture = np.zeros((self.units))

        # Create Toy Dataset

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

    # createTrainSet(self)
    # -----------------------
    # A 4x4 black pixel block simulates a 'car' in a white traffic scene. Every 8
    # pixels is a 'road' that the car can travel horizontally on. Only one car
    # can occupy a road at a time. Each car moves with a speed of 1, 2, or 4
    # pixels at a time, chosen with uniform probability. A car enters an empty
    # row with a probability of 1/8 if there is no car currently there. We use a
    # large dataset with the variation as described above to allow for training
    # to happen and to simulate a real world setting.
 
    def createTrainSet(self):
        
        carColor = .5
        backgroundColor = 1
        trainSet = []
        roads = [0] * 4
        speed = [0] * 4
        position = [0] * 4
        count = 0

        while (count < 10):
            image = np.ones((np.sqrt(self.inputSize),np.sqrt(self.inputSize))) * backgroundColor
            for i in range(0,4):
                if roads[i] == 1:
                    if position[i] < np.sqrt(self.inputSize) - 4:
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
            trainSet.append(np.reshape(image,self.inputSize))
            count += 1
        return trainSet

    def createTestSet(self):        
        
        return []

    def act(self,z):

        return 1/(1 + np.exp(-1 * z))

    def der(self,z):

        return self.act(z) * (1 - self.act(z))

    def cost(self):

        c = []
        for i in range(0,self.outputNum+1):
            dif = self.activations[i] - self.images[i]
            c.append(1.0/2 * np.dot(dif,dif))
        return c

    # forwardProp(self, encoderImages, decoderImages, futureImages)
    # -------------------------------------------------------------
    # Each encoder neuron (i) receives self.inputEncoder[i] as input. This
    # consists of the output of the previous encoder neuron
    # (self.inputEncoderPast[i]) and the weighted image
    # (self.inputImage[i]). The output of each encoder neuron is stored in
    # self.encoder. The first input to the decoder is the output of the last
    # encoder neuron times self.weightBetween. After, each decoder neuron (i)
    # receives self.inputDecoder[i] as input. The output of each decoder neuron
    # is stored in self.decoder. 

    def forwardProp(self,encoderImages,decoderImages,futureImages):

        # Encoder    
        self.encoder = [np.zeros((self.units,self.units))]

        for i in range(0,len(encoderImages)):
            self.inputImage.append(np.dot(self.weightImage,encoderImages[i]) + self.biasImage)
            self.inputEncoderPast.append(np.dot(self.weightEncoder,self.encoder[i]))
            self.inputEncoder.append(self.inputPast[i] + self.inputEncoderPast[i])
            self.encoder.append(self.act(self.inputEncoder[i]))

        self.encoder.remove(0)

        # Decoder
        self.inputDecoder = [np.dot(self.weightBetween,self.encoder[-1]) + self.biasDecoder]
        self.decoder = []

        for i in range(0,len(decoderImages)):
            self.decoder.append(self.act(self.inputDecoder[i]))
            self.inputDecoder.append(np.dot(self.weightDecoder,self.decoder[i]))

        # Future
        self.inputFuture = [np.dot(self.weightBetween,self.encoder[-1]) + self.biasEncoder]
        self.future = []

        for i in range(0,len(futureImages)):
            self.future.append(self.act(self.inputFuture[i]))
            self.inputFuture.append(np.dot(self.weightFuture,self.future[i]))

    # Function: backProp(self, encoderImages, decoderImagesDecoder,
    # imagesFuture):
    # ------------------------------------------------------
    #
    # Decoder and Future
    # ------------------
    # We use a Euclidian loss function: 1/2 ||y - f||^2, where f is the output
    # of our neuron and y is the groundtruth. Let zi be the input into the
    # neuron (i). We first calculate dE/dzi for every neuron. Starting at the right most
    # end of the decoder, we have dE/dz4 = (h - f(z4))*f'(z4). Store this in
    # deltasDecoder[3]. For the next neuron, we have dE/dz3 = d(E' + E'')/dz3,
    # where dE'/dz3 is the error from the immediate groundtruth (I2), calculated just
    # like above. We store this in deltaImageDecoder. E'' is the error from the
    # groundtruth (I1) propogated through the weight (->>). Lets call this
    # weight W. We have d(E'')/dz3 = d(E'')/dz4 * dz3/dz3. Note that z4 =
    # W*f(z3). Thus dz4/dz3 = W*f'(z3). Thus 
    # d(E'')/dz3 = d(E'')/dz4 * W * f'(z3). The first term has already been
    # computed and stored in deltasDecoder[3]. We calculate d(E'')/dz3  and store it
    # in deltaTimeDecoder. Our final dE/dz3 is deltaTimeDecoder +
    # deltaImageDecoder. We store this in deltasDecoder[2]. We will calculate
    # one more for clarity. For the next neuron, we have dE/dz2 = d(E' + E'' +
    # E''')/dz3, where E',E'' are above and E''' is the error from the immediate
    # groundtruth (I3) calculated just like above. We store that in
    # deltaImageDecoder[1]. Note that d(E' + E'')/dz3 = d(E' + E'')/dz2 * dz2/dz3
    # which is the deltaImageDecoder[2] * W * f'(z2). We store d(E' + E'')/dz3
    # in deltaTimeDecoder[1]. We add these deltas together to get deltasDecoder[1].
    # The procedure is exactly the same for the Future module, except we use
    # different images to calculate error (namely, images in the future).
    #
    # Encoder
    # ------------------
    # We look at the right most encoder neuron, (i). Using the same reasoning as
    # above we have, dE/dzi = d(E_decoder)/dzi + d(E_future)/dzi = WBetween *
    # (deltaDecoder[0] + deltaFuture[0]) * f'(zi), where E_decoder and E_future
    # are the sum of errors from the image errors of the decoder and future
    # respectively, i.e E_decoder = E' + E'' + .... We store this total delta in
    # deltasEncoder. Note that each neuron
    # receives input from two sources (its previous time and the image). Thus,
    # dE/dzImage = dE/dzi * dzi/dzImage =  dE/dzi * (dzImage + dzTime)/dzImage =
    # dE/dzi = dE/dzTime. Thus, it is sufficient to only calculate one delta for
    # each neuron in the Encoder. 
    # 
    # Weight Update
    # -------------------
    # Note that for neuron (i) and the weights W entering it, dE/dW = dE/dzi *
    # dzi/dW. Note that zi = W*f(z(i-1)). Thus dzi/dW is the activation of the
    # previous neuron. Thus dE/dW = deltaDecoder[i]*decoder[i-1]. Because our
    # weights are the same, our effective dE/dW = deltaDecoder[1::] *
    # decoder[0:-1]. This also holds for Future. Using
    # the same reasoning, for the weight in between encoder and decoder/future,
    # we have dE/dW =  deltaDecoder[0]*encoder[-1] + deltaFuture[0]*encoder[-1] =
    # (deltaDecoder[0] + deltaFuture[0]) * encoder[-1]. For the encoder weight,
    # we have dE/dW = deltaEncoder[i]*encoder[i-1] and because the encoder
    # weights are the same, our effective dE/dW = deltaEncoder[1::] *
    # encoder[0:-1]. For the input image weight, we have dE/dw =
    # deltaEncoder[i]*imagesEncoder[i]. Thus, because input weights are the
    # same, our effective dE/dW = deltaEncoder*imagesEncoder.
    # 
    # Optimization
    # ------------------
    # Stochastic gradient descent. We multiply derivatives above times a learning rate
    # and subtract it from the current weight. 

    def backProp(self,imagesEncoder,imagesDecoder,imagesFuture):

        # Decoder
        deltasDecoder = [None] * len(imagesDecoder)
        deltasDecoder[-1] = np.dot((self.decoder[-1] - self.imagesDecoder[-1]),self.der(self.inputDecoder[-1]))

        for i in range(0,len(imagesDecoder) - 1)[::-1]:
            deltaTimeDecoder = np.dot(self.weightDecoder.T * deltasDecoder[i+1], self.der(self.inputDecoder[i]))
            deltaImageDecoder = np.dot((self.decoder[i] - self.imagesDecoder[i]),self.der(self.inputDecoder[i]))
            deltasDecoder[i] = deltaImageDecoder + deltaTimeDecoder

        # Future
        deltasFuture = [None] * len(imagesFuture)
        deltasFuture[-1] = np.dot((self.future[-1] - self.imagesFuture[-1]),self.der(self.inputFuture[-1]))

        for i in range(0,len(imagesFuture) - 1)[::-1]:
            deltaTimeFuture = np.dot(self.weightFuture.T * deltasFuture[i+1], self.der(self.inputFuture[i]))
            deltaImageFuture = np.dot((self.future[i] - self.imagesFuture[i]),self.der(self.inputFuture[i]))
            deltasFuture[i] = deltaImageFuture + deltaTimeFuture

        # Encoder
        deltasEncoder = [None] * len(imagesEncoder)
        deltasEncoder[-1] = np.dot(self.weightBetween.T * (deltaDecoder[0] + deltaFuture[0]), self.der(self.inputEncoder[-1]))
                             
        for i in range(0,len(imagesEncoder) - 1)[::-1]:
            deltasEncoder.append(np.dot(self.weightEncoder.T * deltasEncoder[i+1], self.der(self.inputEncoder[i])))

        # Updates
        updateW = np.sum(np.dot(deltasDecoder[1::],self.decoder[0:-1]))
        self.weightDecoder = self.weightDecoder - self.learningRate*updateW

        updateW = np.sum(np.dot(deltasFuture[1::],self.future[0:-1]))
        self.weightFuture = self.weightFuture - self.learningRate*updateW

        updateW = np.sum(np.dot(deltasDecoder[0] + deltasFuture[0],self.encoder[-1]))
        self.weightBetween = self.weightBetween - self.learningRate*updateW

        updateB = deltasDecoder[0]
        self.biasDecoder = self.biasDecoder - self.learningRate*updateB                        
        
        updateB = deltasFuture[0]
        self.biasFuture = self.biasFuture - self.learningRate*updateB                        

        updateW = np.sum(np.dot(deltasEncoder[1::],self.encoder[0:-1]))
        self.weightEncoder = self.weightEncoder - self.learningRate*updateW

        updateW = np.sum(np.dot(deltasImageEncoder,self.imagesEncoder))
        self.weightImage = self.weightInput - self.learningRate*updateW

        updateB = np.sum(deltasImageEncoder)
        self.biasFuture = self.biasFuture - self.learningRate*updateB                        

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

        vid = ani.FuncAnimation(fig,lambda i: plt.imshow(video[i],cmap='gray'),frames=maxFrame,interval=50,repeat=False)
        plt.show()

    def dumpImages(self,images):

        for i in range(0,len(images)):
            plt.imshow(self.reshapeImageWithBorder(images[i]))
            plt.savefig("train_" + str(i) + ".png")

    def run(self):

        if (start < len(self.trainSet) - self.encoderLen - self.futureLen):
            encoderImages = self.trainSet[start,start+self.encoderLen]                 
            decoderImages = self.trainSet[start,start+self.decoderLen]                 
            futureImages = self.trainSet[start,start+self.futureLen]                 
            self.forwardProp(encoderImages, decoderImages, futureImages)
            self.backProp(encoderImages, decoderImages, futureImages)
            print self.cost()


        
