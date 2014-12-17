class Layer:
    def __init__(self, units,channels):
        self.units = units
        self.channels = channels

class Parameters:
    def __init__(self, layers, alpha, momentum, anneal, epochs, numImages):
        self.layers = layers
        self.alpha = alpha
        self.momentum = momentum
        self.anneal = anneal
        self.epochs = epochs
        self.numImages = numImages

