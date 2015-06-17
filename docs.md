# Documentation

## Overview

This neural network architecture is built upon "Unsupervised Learning of Video Representations using LSTMs" by Srivastava, et al. Let 'N' represent a sigmoid layer/unit, 'I0, I1, ... ' represent images, and '^','->','->>','>>>','-->','|','||' all represent different weights and connections. The following is a visual representation of this network for a sequence of three images without bias terms. The output of each decoder and future unit is compared against groundtruth. 
 
     Encoder:          Decoder:

      N -> N -> N -> N --> N ->> N ->> N ->> N
      ^    ^    ^    ^     |     |     |     |
      I0   I1   I2   I3    I3    I2    I1    I0

                       Future:

                       --> N >>> N >>> N >>> N
                           ||    ||    ||    ||
                           I4    I5    I6    I7

Built on python, numpy, & cudamat.

## Structure Names

Note, 'In' refers to input, 'Im' refers to image, 'enc' refers to encoder, 'dec' refers to decoder, 'fut' refers to future, 'W' refers to weight, and 'B' refers to bias. 

- *encOut[:,[i]]*: the output of an encoder layer at time i
- *encIn[:,[i]]*: the input into an encoder layer at time i
- encInIm[:,[i]]: the input into an encoder layer from the original image (i.e. encImW * image(i) + encImB)
encInPast[:,[i]]: the input into an encoder layer from the previous time step (i.e. encOut[:,[i-1]])
encImW: the weight matrix an input image is multiplied by
encImB: the bias that gets added to the encoder input
encW: the weight matrix encoder output is multiplied by
betW: the weight matrix the encoder output is multiplied by before going into the decoder/future
decB: the bias that gets added to the encoder output before going into the decoder
futB: the bias that gets added to the encoder output before going into the decoder

## Implementation Notes

To make debugging easier, we treat go to extra pains to make every structure in column form, to fit the math. For example, we will write encOut[:,[i]] instead of encOut[:,i] because we want numpy.shape to return a column vector instead of an array. 

We use matplotlib to view videos. This seems to have some lag, but is effective for debugging.

## Training Set

createTrainingSet(self) creates a toy dataset for us to train with. A 4x4 black pixel block simulates a 'car' in a white traffic scene. Every 16 pixels is a 'road' that the car can travel horizontally on. Only one car can occupy a road at a time. Each car moves with a speed of 1, 2, or 4 pixels at a time, chosen with uniform probability. A car enters an empty row with a probability of 1/4 if there is no car currently there. We use a large dataset with the variation as described above to allow for training to happen and to simulate a real world setting.

## calculateLoss(self,decImages,futImTruth)

Loss with current network. Requires having done one forward prop.

## forwardProp(self, encoderImages, decoderImages, futureImages)

     Each encoder neuron (i) receives self.encIn[i] as input. This
     consists of the output of the previous encoder neuron
     (self.encInPast[i]) and the weighted image
     (self.encInIm[i]). The output of each encoder neuron is stored in
     self.encOut. The first input to the decoder is the output of the last
     encoder neuron times self.encDecFutW plus a bias. After, each decoder neuron (i)
     receives self.decIn[i] as input. The output of each decoder neuron
     is stored in self.dec. This is multiplied by self.decImW
     to create a decoded image. Future works similarly. We calculate loss at
     the end

## backProp(self, encoderImages, decoderImagesDecoder, imagesFuture):
    
     Decoder and Future
     ------------------
     We use a Euclidian loss function: 1/2 ||y - f||^2, where f is the output
     of our neuron and y is the groundtruth. Let zi be the input into the
     neuron (i). We first calculate dE/dzi for every neuron. Starting at the right most
     end of the decoder, we have dE/dz4 = (h - f(z4))*f'(z4). Store this in
     deltasDecoder[3]. For the next neuron, we have dE/dz3 = d(E' + E'')/dz3,
     where dE'/dz3 is the error from the immediate groundtruth (I2), calculated just
     like above. We store this in deltaImageDecoder. E'' is the error from the
     groundtruth (I1) propogated through the weight (->>). Lets call this
     weight W. We have d(E'')/dz3 = d(E'')/dz4 * dz3/dz3. Note that z4 =
     W*f(z3). Thus dz4/dz3 = W*f'(z3). Thus 
     d(E'')/dz3 = d(E'')/dz4 * W * f'(z3). The first term has already been
     computed and stored in deltasDecoder[3]. We calculate d(E'')/dz3  and store it
     in deltaTimeDecoder. Our final dE/dz3 is deltaTimeDecoder +
     deltaImageDecoder. We store this in deltasDecoder[2]. We will calculate
     one more for clarity. For the next neuron, we have dE/dz2 = d(E' + E'' +
     E''')/dz3, where E',E'' are above and E''' is the error from the immediate
     groundtruth (I3) calculated just like above. We store that in
     deltaImageDecoder[1]. Note that d(E' + E'')/dz3 = d(E' + E'')/dz2 * dz2/dz3
     which is the deltaImageDecoder[2] * W * f'(z2). We store d(E' + E'')/dz3
     in deltaTimeDecoder[1]. We add these deltas together to get deltasDecoder[1].
     The procedure is exactly the same for the Future module, except we use
     different images to calculate error (namely, images in the future).
    
     Encoder
     ------------------
     We look at the right most encoder neuron, (i). Using the same reasoning as
     above we have, dE/dzi = d(E_decoder)/dzi + d(E_future)/dzi = WBetween *
     (deltaDecoder[0] + deltaFuture[0]) * f'(zi), where E_decoder and E_future
     are the sum of errors from the image errors of the decoder and future
     respectively, i.e E_decoder = E' + E'' + .... We store this total delta in
     deltasEncoder. Note that each neuron
     receives input from two sources (its previous time and the image). Thus,
     dE/dzImage = dE/dzi * dzi/dzImage =  dE/dzi * (dzImage + dzTime)/dzImage =
     dE/dzi = dE/dzTime. Thus, it is sufficient to only calculate one delta for
     each neuron in the Encoder. 
     
     Weight Update
     -------------------
     Note that for neuron (i) and the weights W entering it, dE/dW = dE/dzi *
     dzi/dW. Note that zi = W*f(z(i-1)). Thus dzi/dW is the activation of the
     previous neuron. Thus dE/dW = deltaDecoder[i]*decoder[i-1]. Because our
     weights are the same, our effective dE/dW = deltaDecoder[1::] *
     decoder[0:-1]. This also holds for Future. Using
     the same reasoning, for the weight in between encoder and decoder/future,
     we have dE/dW =  deltaDecoder[0]*encoder[-1] + deltaFuture[0]*encoder[-1] =
     (deltaDecoder[0] + deltaFuture[0]) * encoder[-1]. For the encoder weight,
     we have dE/dW = deltaEncoder[i]*encoder[i-1] and because the encoder
     weights are the same, our effective dE/dW = deltaEncoder[1::] *
     encoder[0:-1]. For the input image weight, we have dE/dw =
     deltaEncoder[i]*imagesEncoder[i]. Thus, because input weights are the
     same, our effective dE/dW = deltaEncoder*imagesEncoder.
     
     Optimization
     ------------------
     Stochastic gradient descent. We multiply derivatives above times a learning rate
     and subtract it from the current weight. 

## Image and Video Processing

     reshapeImageWithBorder: reshapes image vector to array and adds a black
     background. Without one black and white pixel, matplotlib doesn't know the
     range of grayscale. 
     viewImage: shows grayscale image
     viewVideo: uses matplotlib animation to show images in video form
     dumpImages: saves images as PNGs


##  run(self)
  
     Runs forward, backprop and training of network. Encodes self.encLen
     images, decodes self.decodeLen images, and predicts self.futLen images
     of the future. Currently does one pass through trainSet. 

