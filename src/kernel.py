import numpy as np

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

# Returns the weight errors, bias error, and previous layer errors given a 2D image
# Args:
#   in_shape (tuple) - (original image height, original image length)
#   out_shape (tuple) - (filtered image height, filtered image length)
#   zs (2D np arr) - unsquashed activations of original image
#   weights (2D np arr) - weights of a kernel
#   bias (float) - kernel bias
#   deltas (2D np arr) - forward errors (with out_shape shape)
def prev_errors (in_shape, out_shape, zs, weights, bias, deltas):
    deltasPrev = np.zeros(in_shape)
    kernelHeight = len(weights)
    kernelLength = len(weights[0])

    weightDeltas = np.zeros((kernelLength,kernelHeight))
    biasDelta = 0

    # Loop for each step
    for y in range(out_shape[0]):
        for x in range(out_shape[1]):
            d = deltas[y][x]

            # Loop for each part of the kernel
            for wy, dpy in enumerate(range(y, y+kernelHeight)):
                for wx, dpx in enumerate(range(x, x+kernelLength)):
                    deltasPrev[dpy][dpx] += d*weights[wy][wx]*func_deriv(zs[dpy][dpx])

    # Loop kernel across image to calculate grad_w
    for y in range(in_shape[0]-kernelHeight+1):
        for x in range(in_shape[1]-kernelLength+1):
            # Calculate for one convolution
            for wy in range(kernelHeight):
                for wx in range(kernelLength):
                    weightDeltas[wy][wx] += func(zs[y+wy][x+wx])*deltasPrev[y+wy][x+wx]

    for dp in deltasPrev:
        for d in dp:
            biasDelta+=d

    return weightDeltas, biasDelta, deltasPrev

# Individual kernel objects
class Kernel:
    # Args:
    #   filter_size: a 3-tuple (kernel depth, kernel height, kernel length)
    def __init__(self, filter_size, weights=None, bias=None):
        self.filter_size = filter_size
        self.feature_map_length = filter_size[2]
        self.feature_map_height = filter_size[1]
        self.num_feature_maps = filter_size[0]
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [np.random.randn(self.feature_map_height, self.feature_map_length) for f in range(self.num_feature_maps)]

        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.random()

    # Takes in a list of images and applies the filter specific to the object to the filter, returning the new 2D image
    # Args:
    #   image_list: a list of 2D images
    def use_kernel (self, image_list):
        num_images = len(image_list)
        new_image_size = (len(image_list[0]) - self.feature_map_height + 1, len(image_list[0][0]) - self.feature_map_length + 1)
        new_image = np.zeros(new_image_size)
        for i in range(num_images):
            new_image += self.use_feature_map(self.weights[i], image_list[i], new_image_size)
        for y in range(new_image_size[0]):
            for x in range(new_image_size[1]):
                new_image[y][x] = new_image[y][x]+ self.bias
        return new_image

    # Returns the weight, bias, and delta errors given a current set of deltas
    # Args:
    #   input_image_shape (3-tuple) - a 3 tuple for the shape of input images (num images, image height, image length)
    #   output_image_shape (3-tuple) - a 3 tuple for the shape of the output images (same format)
    #   z-activations (3D np array) - the previous non-squashed activations
    #   deltas (2D np array) - the previous errors (for the single kernel)
    def get_errors(self, input_image_shape, output_image_shape, z_activations, deltas):
        deltaPrevs = []
        weightErrors = []
        biasError = 0

        for w, z in zip(self.weights, z_activations):
            w_err, b_err, d_err = prev_errors(input_image_shape[1:],output_image_shape[1:],
                                  z,w,self.bias,deltas)
            deltaPrevs.append(d_err)
            weightErrors.append(w_err)
            biasError += b_err

        return weightErrors, biasError, deltaPrevs

    # This method takes in a feature map and slides it across an image
    # Returns:
    #   a 2D array which is the new output image
    def use_feature_map (self, feature_map, image, new_image_size):
        new_image = np.zeros(new_image_size)
        for x in range(new_image_size[1]):
            for y in range(new_image_size[0]):
                img_piece = image[y:y+self.feature_map_height,x:x+self.feature_map_length]
                print(np.dot(feature_map.ravel(), img_piece.ravel()))
                new_image[y][x] = np.dot(feature_map.ravel(), img_piece.ravel())
        return new_image

    # Updates the kernels weights and biases
    #   weight_update (3D np arr) - what to add to the weights
    #   bias_update (float) - what to add to the bias
    def update (self, weight_update, bias_update):
        self.weights += weight_update
        self.bias += bias_update

    def set_weights(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias
