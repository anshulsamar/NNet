import tools

layers = [Layer(100,1),Layer(10,2),Layer(100,1)]
alpha = 1e-4
momentum = 0.9
anneal = 1.2
epochs = 10
numImages = 10000
data = ""

nn = NNet(Parameters(layers,alpha,momentum,anneal,epochs))
nn.run()
