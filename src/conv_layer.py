import numpy as np
from kernel import Kernel

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

class ConvLayer:
    # Args:
    #   image_shape (3 tuple (ints)) - (image depth, image height, image length)
    #   kernel_shape (4 tuple (ints)) - (num kernels, kernel depth, kernel height, kernel length)
    def __init__(self, image_shape, kernel_shape, kernels=None):
        self.image_shape = image_shape
        self.kernel_shape = kernel_shape
        self.output_shape = (kernel_shape[0], image_shape[1]-kernel_shape[2]+1, image_shape[2]-kernel_shape[3]+1)

        if kernels is not None:
            self.kernels = kernels
        else:
            self.kernels = []
            for x in range(kernel_shape[0]):
                self.kernels.append(Kernel(kernel_shape[1:]))

    def get_kernels(self, index=-1):
        if index == -1:
            return self.kernels
        return self.kernels[index]


    # Similar to feedforward, but without squashing
    # Args: image - 3D np array of the image
    def get_activations(self, image):
        new_img = []
        for k in self.kernels:
            new_img.append(k.use_kernel(image))
        return np.array(new_img)

    # Returns the new image created using the current layers kernels squashed by an activation function
    # Args: image - 3D np array of the image
    def feed_forward(self, image):
        new_img = self.get_activations(image)
        return func(new_img)


    # Returns the kernel errors (weights and biases) and the previous image error
    # Args:
    #   z-activations - activations for the previous layer
    #   deltas - a 3D np array of the errors in the forward layer
    def backprop (self, z_activations, deltas):
        prevDeltas = []
        kernelWeightDeltas = []
        kernelBiasDeltas = []

        for k, d in zip(self.kernels, deltas):
            wd, bd, pd = k.get_errors(self.image_shape, self.output_shape, z_activations, d)
            prevDeltas.append(pd)
            kernelWeightDeltas.append(wd)
            kernelBiasDeltas.append(bd)

        return np.array(kernelWeightDeltas), np.array(kernelBiasDeltas), np.array(prevDeltas)

    # Update the kernels
    # Args:
    #   d_weights - 4d np array to change kernel weights
    #   d_bias - 1d np array to change kernel bias
    def update (self, d_weights, d_bias):
        for i, k in enumerate(self.kernels):
            k.update(d_weights[i], d_bias[i])

    def get_output_shape(self):
        return self.output_shape
