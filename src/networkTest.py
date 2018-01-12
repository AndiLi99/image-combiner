import mnist_loader
import network2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
# net = network2.Network([784, 70, 784], cost=network2.CsrossEntropyCost)

net = network2.load("../trained_networks/mnist_network.txt")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.03, evaluation_data=validation_data, lmbda = 5.0)
#
# print net.total_cost(training_data, 0)
# net.save("demo_network.txt")
# net = network2.load("demo_network.txt")
#
# # # img = net.feedforward(training_data[0][0])
plt.imsave("demo_99.png", training_data[53][0].reshape(28,28), cmap=cm.gray)
plt.imsave("demo_.png", training_data[15][0].reshape(28,28), cmap=cm.gray)

#
# print net.total_cost(training_data, 0)

for i in range(1):
    img = network2.transform(training_data[53][0], training_data[15][0], (1.0*i)/45, net)
    string = "demo_" + str(i) + ".png"
    plt.imsave(string, img.reshape(28,28), cmap=cm.gray)
