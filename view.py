import numpy as np
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nnet
import dataLoader

def reshapeImageWithBorder(image):

    dim = np.sqrt(len(image))
    array = np.reshape(image,(dim,dim))
    arrayBorder = np.zeros((dim+2,dim+2))
    arrayBorder[1:1+dim,1:1+dim] = array
    return arrayBorder

def viewImage(images,i):
    plt.imshow(reshapeImageWithBorder(images[:,[i]]),cmap = 'gray')
    plt.show()

def viewVideo(images,maxFrame):

    fig = plt.figure()
    video = []
    for i in range(0,np.shape(images)[1]):
        video.append(reshapeImageWithBorder(images[:,[i]]))

    vid = ani.FuncAnimation(fig,lambda i: plt.imshow(video[i],\
              cmap='gray'),frames=maxFrame,interval=1,repeat=False)
    plt.show()

def viewOutput(net,fileNum,frame=None):

    [decIm,futIm] = dataLoader.loadOutput(net,fileNum)
    trainSet = dataLoader.loadTrainingSet(net,fileNum)

    video = []
    for i in range(0,net.imPerFile):
        dim = np.sqrt(net.imSize)
        enc = np.reshape(trainSet[:,[i]],(dim,dim))
        dec = np.reshape(decIm[:,[net.decLen-i-1]],(dim,dim))
        futTruth = np.reshape(trainSet[:,[i+net.encLen]],(dim,dim))
        fut = np.reshape(futIm[:,[i]],(dim,dim))
        array = np.zeros((2*dim+4,2*dim+4))
        array[1:1+dim,1:1+dim] = enc
        array[1:1+dim,2+dim:2+2*dim] = futTruth
        array[2+dim:2+2*dim,1:1+dim] = dec
        array[2+dim:2+2*dim,2+dim:2+2*dim] = fut
        video.append(array)
        
    if (frame != None):
        plt.imshow(video[frame],cmap = 'gray')
    else:
        fig = plt.figure()
        vid = ani.FuncAnimation(fig,lambda i: plt.imshow(video[i],\
              cmap='gray'),frames=net.imPerFile,interval=1,repeat=False)
    plt.show()

def dumpImages(images):

    for i in range(0,len(images)):
        plt.imshow(reshapeImageWithBorder(images[i]))
        plt.savefig("train_" + str(i) + ".png")


    
