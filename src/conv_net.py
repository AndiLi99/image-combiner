import numpy as np

class Network:
    """ filter_sizes is a list of 3-tuple (kernel_size, padding, stride)
        layer_types is a list of each layer indicating the layer type
        "conv", or "full"
        image_channels is number of channels describing a picture
        ex: RGB is 3 channels
    """
    def __init__ (self, image_channels, filter_sizes, layer_types, MaxPooling=True):
        self.layer_types = layer_types
        self.filter_sizes = filter_sizes
        # filter is a list of tuples (weight, bias)
        self.layers = []


    def large_weight_initializer(self):

    """ input is a 3 dimensional array (z, x, y)
        where z is the dimension of layers of images """
    def feed_forward(self, input_layer):


    #image is a 2D image, filter is a tuple (weight, bias)
    def activate_filter(self, image, filter):

#returns a padded image with padding of padding
def pad_input (image, padding):
    img = np.zeros((image.size[0] + 2*padding, image.size[1] + 2*padding))
    return img
