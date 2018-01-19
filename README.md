# Update
The autoencoder network is now fully convolutional, with deconvolutional layers used in order to reconstruct images. Included is now a dataset of faces in which the network is being trained on. The faces are of size (250x250), and can be accessed through the face_loader.py script, which will load the images into a 3D vector for input. There is also a class that is able to read URL's and automatically webscrape images.

# image-combiner
Autoencoder neural network built for combining images by encoding two separate images together and decoding the result.

The neural network is trained on a dataset consisting of the same input as output. The structure of the neural network contains a bottle-neck, a significantly lower amount of neurons in the middle layer. This forces the neural network to learn to encode and decode the image using the first and second halves of the network respectively.

In our training example we used a 784 -> 30 -> 784 fully connected neural network in order to process images. 

Currently the network is trained to combine images from the MNIST dataset. In the future we will apply this to various other types of images (such as faces, art, photographs)

# Demo

![Demo.gif](https://media.giphy.com/media/l49JDWvO9kGd2RduM/giphy.gif)
This is one example of the combination of two mnist images. Here the number transforms from a 4 to a 7.

# Acknowledgements
The fully connected neural network part of this project makes use of a heavily modified version of Michael Nielsen's neural network code. His code and book can be found at http://neuralnetworksanddeeplearning.com
