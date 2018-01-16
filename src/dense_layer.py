import numpy as np

# Makes a 3D np array into a 1D np array
def flatten_image(image):
    l = np.array([])
    for x in image:
        l = np.concatenate((l, x.ravel()))

    image = l.ravel()
    return image


# leaky relu function
def func (z):
    if isinstance(z, float) or isinstance(z, int):
        if z > 0:
            return z
        else:
            return 0.1*z

    for i, zi in enumerate(z):
        z[i] = func(zi)
    return z

def func_deriv(z):
    if isinstance(z, float) or isinstance(z, int):
        if z > 0:
            return 1
        else:
            return 0.1

    for i, zi in enumerate(z):
        z[i] = func_deriv(zi)
    return z

class DenseLayer:
    # Args:
    #   layer_shape - a 2-tuple of ints (number of neurons on current layer, number of neurons on previous layer)
    #   weights (optional) - a 2D np array of the weights
    #   biases (optional) a 1D np array of the biases
    def __init__(self, layer_shape, weights=None, biases=None):
        self.layer_shape = layer_shape
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.randn(layer_shape[0], layer_shape[1])

        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.random.randn(layer_shape[0])

    # Similar to feed forward but without squashing
    def get_activations(self, input_activations):
        return np.dot(self.weights, input_activations) + self.biases

    # Feeds the input through the layer and uses leaky relu as an logistic function
    # Args:
    #   input_activations - a 1D np array of the previous activations
    def feed_forward(self, input_activations):
        return func(self.get_activations(input_activations))

    # Returns the gradients for the weights, biases, and the deltas for the previous layer
    def backprop (self, z_activations, deltas):
        if len(z_activations.shape) == 3:
            z_activations = flatten_image(z_activations)
        prevDeltas = np.dot(self.weights.transpose(), deltas) * func_deriv(z_activations)
        biasDeltas = deltas
        weightDeltas = np.dot(np.array([deltas]).transpose(), np.array([func(z_activations)]))

        return weightDeltas, biasDeltas, prevDeltas

    # Updates layers parameters
    # Args:
    #   d_weights - 2D np array determining how much to change the weights by
    #   d_biases - 1D np array determining how much to change the biases by
    def update(self, d_weights, d_biases):
        self.weights += d_weights
        self.biases += d_biases

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases