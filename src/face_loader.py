import tarfile
import os
import os.path
import imageio
import numpy as np

def load_data():
    f = tarfile.open("../data/lfw-deepfunneled.tgz")

def load_data_wrapper():
    training_data = []
    for dirpath, dirnames, filenames in os.walk("../data/lfw-deepfunneled"):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            imgdir = os.path.join(dirpath, filename)
            img = imageio.imread(imgdir)
            print imgdir
            training_data.append(img)
    return training_data

print np.shape(load_data_wrapper())
