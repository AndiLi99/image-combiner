import numpy as np
from conv_layer import ConvLayer
from kernel import Kernel
from dense_layer import DenseLayer
from random import shuffle
from copy import deepcopy

# Makes a 3D np array into a 1D np array
def flatten_image(image):
    l = np.array([])
    for x in image:
        l = np.concatenate((l, x.ravel()))

    image = l.ravel()
    return image

# Makes 1D np array into 3D np array
def convert_to_image(arr, image_shape):
    image = np.zeros(image_shape)
    counter = 0

    for z in range(image_shape[0]):
        for y in range(image_shape[1]):
            for x in range(image_shape[2]):
                image[z][y][x] = arr[counter]
                counter+=1

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

# Derivative for leaky relu
def func_deriv(z):
    if isinstance(z, float) or isinstance(z, int):
        if z > 0:
            return 1
        else:
            return 0.1

    for i, zi in enumerate(z):
        z[i] = func_deriv(zi)
    return z

class QuadCost:
    @staticmethod
    def cost (network_output, expected_output):
        return sum(0.5*(np.power(network_output-expected_output, 2)))

    @staticmethod
    def delta (network_output, z_activation_deriv, expected_output):
        return 0.5*(np.power(network_output-expected_output, 2)*z_activation_deriv)


class Convolutional:
    # Args:
    #   layer_types - (list) a list of strings indicating the layer type. "conv" or "dense"
    #   layer_shapes - (list of list of tuples) a list of the shapes for each layer
    #                       conv layers - [image shape, kernel shape]
    #                       dense layers - [layer shape]
    def __init__(self, layer_types, layer_shapes, layers=None, cost_func=QuadCost):
        self.layer_types = layer_types
        self.layer_shapes = layer_shapes
        self.num_layers = len(layer_types)
        self.cost_func = cost_func
        if layers is not None:
            self.layers = layers
        else:
            self.layers = []
            for lt, ls in zip(layer_types, layer_shapes):
                if lt == "conv":
                    self.layers.append(ConvLayer(image_shape=ls[0], kernel_shape=ls[1]))
                elif lt == "dense":
                    self.layers.append(DenseLayer(layer_shape=ls[0]))

    # Returns the next activation without squashing it
    # Args:
    #   z_activations - (np arr) the current activations
    #   layer - the next layer to be used
    def next_activation(self, z_activations, layer):
        return layer.get_activations(z_activations)

    # Feeds an input through the network, returning the output
    # Args: network_input - (np arr) the input
    def feed_forward(self, network_input):
        is_conv = False
        if self.layer_types[0] == "conv":
            is_conv = True
            if len(network_input.shape) == 2:
                network_input = np.array([network_input])


        for lt, lyr in zip(self.layer_types, self.layers):
            # Squash to 1D np array
            if lt is not "conv" and is_conv:
                is_conv = False
                network_input = flatten_image(network_input)

            network_input = lyr.feed_forward(network_input)

        return network_input

    # This function calculates the gradients for one training example
    # Args:
    #   network_input - (np arr) the input being used
    #   expected_output - (np arr) the expected output
    def backprop(self, network_input, expected_output):
        curr_z = network_input
        z_activations = [network_input]

        is_conv = False
        if self.layer_types[0] is "conv":
            is_conv = True

        for lt, lyr in zip(self.layer_types, self.layers):
            # Squash to 1D np array
            if lt is not "conv" and is_conv:
                is_conv = False
                curr_z = flatten_image(curr_z)

            curr_z = lyr.get_activations(curr_z)
            z_activations.append(deepcopy(curr_z))
            curr_z = func(curr_z)

        # Store derivatives and activation for output layer
        squashed_activations_deriv = func_deriv(deepcopy(curr_z))
        squashed_activations = curr_z

        # Errors for the last layer
        delta = self.cost_func.delta(squashed_activations,
                                     squashed_activations_deriv,
                                     expected_output)

        is_conv = True
        if self.layer_types[self.num_layers-1] is not "conv":
            is_conv = False

        delta_w = []
        delta_b = []

        # Append all the errors for each layer
        for lt, lyr, zprev in reversed(zip(self.layer_types, self.layers, z_activations[:-1])):
            if lt is "conv" and not is_conv:
                delta = convert_to_image(delta, lyr.get_output_shape())
                is_conv = True

            dw, db, dlt = lyr.backprop(zprev, delta)
            delta_w.insert(0, dw)
            delta_b.insert(0, db)

            delta = dlt

        return np.array(delta_w), np.array(delta_b)

    # Updates the network given a specific minibatch (done by averaging gradients over the minibatch)
    # Args:
    #   mini_batch - a list of tuples, (input, expected output)
    #   step_size - the amount the network should change its parameters by relative to the gradients
    def update_network(self, mini_batch, step_size):
        gradient_w, gradient_b = self.backprop(mini_batch[0][0], mini_batch[0][1])

        for inp, outp in mini_batch[1:]:
            dgw, dgb = self.backprop(inp, outp)
            gradient_w += dgw
            gradient_b += dgb

        # Average the gradients
        gradient_w *= step_size/(len(mini_batch)+0.00)
        gradient_b *= step_size/(len(mini_batch)+0.00)

        # Update weights and biases in opposite direction of gradients
        for gw, gb, lyr in zip(gradient_w, gradient_b, self.layers):
            lyr.update(-gw, -gb)

    # Evaluates the average cost across the training set
    def evaluate_cost(self, training_set):
        total = 0.0
        for inp, outp in training_set:
            net_outp = self.feed_forward(inp)
            total += self.cost_func.cost(net_outp, outp)
        return total/len(training_set)

    # Performs SGD on the network
    # Args:
    #   epochs - (int), number of times to loop over the entire batch
    #   step_size - (float), amount network should change its parameters per update
    #   mini_batch_size - (int), number of training examples per mini batch
    #   training_inputs - (list), the list of training inputs
    #   expected_outputs - (list), the list of expected outputs for each input
    def stochastic_gradient_descent(self, epochs, step_size, mini_batch_size, training_inputs, expected_outputs):
        training_set = []
        for inp, outp in zip(training_inputs, expected_outputs):
            training_set.append((inp, outp))

        # Train
        for ep in range(epochs):
            shuffle(training_set)
            for x in range(0, len(training_set), mini_batch_size):
                self.update_network(training_set[x:x+mini_batch_size], step_size)
            # Update with progress
            print("Epoch: %d   Average cost: %f" % (ep+1, self.evaluate_cost(training_set)))
