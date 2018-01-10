# image-combiner
Autoencoder neural network built for combining images by encoding two separate images together and decoding the result.

The neural network is trained on a dataset consisting of the same input as output. The structure of the neural network contains a bottle-neck, a significantly lower amount of neurons in the middle layer. This forces the neural network to learn to encode and decode the image using the first and second halves of the network respectively.

In our training example we used a 784 -> 30 -> 784 fully connected neural network in order to process images. 

Currently the network is trained to combine images from the MNIST dataset. In the future we will apply this to various other types of images (such as faces, art, photographs)

# Demo













# Acknowledgements
This project makes use of a heavily modified version of Michael Nielsen's neural network code. His code and book can be found at http://neuralnetworksanddeeplearning.com
