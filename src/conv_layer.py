import numpy as np
import kernel

class ConvLayer:
    #   filter_size: a 4-tuple (num_kernels, kernel depth, kernel height, kernel length)
    #   weights: list of weights for each kernel
    #   biases: list of biases for each kernel
    def __init__ (self, kernel_size, weights=None, biases=None):
        self.kernels = []
        self.kernel_size = kernel_size
        for i in range(kenrel_size[0]):
            if weights is not None:
                kernels.append(kernels.Kernel(kernel_size[1:], weights[i], biases[i]))
            else:
                kernels.append(kernels.Kernel(kernel_size[1:]))


    #   input_layer is a the 3D array (z, y, x)
    def feedforward (self, input_layer):
        output_layer = []
        for i in range(self.kernel_size[0]):
            output_layer.append(self.kernels[i].use_kernel(input_layer))
        return output_layer

    # Returns a list of weight, bias, and delta errors given a current set of deltas for each kernel
    # Args:
    #   input_image_shape (3-tuple) - a 3 tuple for the shape of input images (num images, image height, image length)
    #   output_image_shape (3-tuple) - a 3 tuple for the shape of the output images (same format)
    #   z-activations (3D np array) - the previous non-squashed activations
    #   deltas (3D np array) - list of the previous errors
    def get_errors (self, input_shape, output_shape, z_activations, deltas):
        errors = []
        for i in range(self.kernel_size[0]):
            errors.append(self.kernels[i].get_errors(input_shape, output_shape, z_activations, deltas[i]))
        return sum(errors)

    def update (self, weight_update, bias_update):
        for i in range(self.kernel_size[0]):
            self.kernels[i].update(weight_update[i], biase_update[i])

    def set_weights(self, weight, index):
        self.kernels[index].set_weights (weight)

    def set_biases(self, bias, index):
        self.kernels[index].set_bias(bias)
