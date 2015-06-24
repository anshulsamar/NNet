import numpy as np
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class view(object):

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

    def viewOutput(self,fileNum,frame=None):

        [decIm,futIm] = self.loadOutput(fileNum)
        trainSet = self.loadTrainingSet(fileNum)

        video = []
        for i in range(0,self.imPerFile):
            dim = np.sqrt(self.imSize)
            enc = np.reshape(trainSet[:,[i]],(dim,dim))
            dec = np.reshape(decIm[:,[self.decLen-i-1]],(dim,dim))
            futTruth = np.reshape(trainSet[:,[i+self.encLen]],(dim,dim))
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
              cmap='gray'),frames=self.imPerFile,interval=1,repeat=False)
        plt.show()

    def dumpImages(self,images):
    
        for i in range(0,len(images)):
            plt.imshow(self.reshapeImageWithBorder(images[i]))
            plt.savefig("train_" + str(i) + ".png")

