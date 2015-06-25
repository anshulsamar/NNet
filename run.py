import pickle
import nnet


def saveNN(nn,saveNNFile):
    f = open(saveNNFile,'w')
    pickle.dump(nn,f)
    f.close()

def loadNN(saveNNFile):
    f = open(saveNNFile,'r')
    return pickle.load(f)

def runNN(load):

    saveNNFile = 'backup.p'

    if (load):
        a = loadNN(saveNNFile)
    else:
        a = nnet.nnet()

    for e in range(a.currEpoch,10):
        a.currEpoch = e
        a.train()
        saveNN(a,saveNNFile)
            
        
    
