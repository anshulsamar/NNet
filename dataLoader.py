import numpy as np
import nnet
import pickle

def createTrainingSet(net):

    carColor = .5
    backgroundColor = 1
    roads = [0] * 4
    speed = [0] * 4
    position = [0] * 4
    count = 0
    fileCount = 0

    while (fileCount < net.numDataFiles):

        trainSet = np.zeros((net.imSize,0))
        trainSetNext = np.zeros((net.imSize,0))
        count = 0

        while (count < net.imPerFile + net.futLen):
            dim = np.sqrt(net.imSize)
            image = np.ones((dim,dim)) * backgroundColor
            for i in range(0,4):
                if roads[i] == 1:
                    if position[i] < np.sqrt(net.imSize) - 4:
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
            reshaped = np.reshape(image,(net.imSize,1))
            trainSet = np.hstack((trainSet,reshaped))
            count += 1

        dFile = open(net.dataFile[:-2] + str(fileCount) + '.p','w')
        pickle.dump(trainSet,dFile)
        dFile.close()
        fileCount += 1
     
def loadTrainingSet(net,fileNum):

    trainSet = np.zeros((net.imSize,0))
    trainFile = open(net.dataFile[:-2] + str(fileNum) + '.p','rb')
    
    while 1:
        try:
            trainSet = np.hstack((trainSet,pickle.load(trainFile)))
        except (EOFError):
            break

    trainFile.close()
    return trainSet

def loadOutput(net, fileNum):

    decIm = np.zeros((net.imSize,0))
    decFile = open(net.decOutFile[:-2] + str(fileNum) + '.p','rb')

    while 1:
        try:
            decIm = np.hstack((decIm,pickle.load(decFile)))
        except (EOFError):
            break

    decFile.close()

    futIm = np.zeros((net.imSize,0))
    futFile = open(net.futOutFile[:-2] + str(fileNum) + '.p','rb')

    while 1:
        try:
            futIm = np.hstack((futIm,pickle.load(futFile)))
        except (EOFError):
            break

    futFile.close()

    return [decIm,futIm]





    
