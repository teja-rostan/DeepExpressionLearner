"""
[Model program]
"""
import nnet
import numpy as np
import theano
from theano import tensor as T


class CNNLearner:

    def __init__(self, class_size, n_hidden_neurons=30, conv_type="class"):
        """
        Initialization of Classification neural network.
        :param attribute_size: Number of input attributes for neural network.
        :param class_size: Number of output classes for neural network.
        :param n_hidden_neurons: Number of hidden neurons in every hidden layer in neural network architecture.

        """
        self.class_size = class_size
        self.n_hidden_neurons = n_hidden_neurons

        self.n_kernels = 32
        self.k = 6
        self.final_image_size = 73728  # manual fast fix

        X = T.ftensor4()
        Y = T.fmatrix()

        self.w_h = nnet.init_weights((self.n_kernels, 1, 1, self.k * 4))
        self.w_h2 = nnet.init_weights((self.n_kernels * 2, self.n_kernels, 1, self.k))
        self.w_h3 = nnet.init_weights((self.n_kernels * 4, self.n_kernels * 2, 1, self.k))
        self.w_h4 = nnet.init_weights((self.final_image_size, self.n_hidden_neurons))
        self.w_h5 = nnet.init_weights((self.n_hidden_neurons, self.n_hidden_neurons))
        self.w_o = nnet.init_weights((self.n_hidden_neurons, self.class_size))

        if conv_type == "reg":
            self.noise_py_x = nnet.conv_model_reg(X, self.w_h, self.w_h2, self.w_h3, self.w_h4, self.w_h5, self.w_o, 0.03, 0.1)
            self.py_x = nnet.conv_model_reg(X, self.w_h, self.w_h2, self.w_h3, self.w_h4, self.w_h5, self.w_o, 0., 0.)
            self.cost = nnet.rmse(self.noise_py_x, Y)
        elif conv_type == "class":
            self.noise_py_x = nnet.conv_model(X, self.w_h, self.w_h2, self.w_h3, self.w_h4, self.w_h5, self.w_o, 0.03, 0.1)
            self.py_x = nnet.conv_model(X, self.w_h, self.w_h2, self.w_h3, self.w_h4, self.w_h5, self.w_o, 0., 0.)

            self.cost = T.mean(T.nnet.categorical_crossentropy(self.noise_py_x, Y))
        params = [self.w_h, self.w_h2, self.w_h3, self.w_h4, self.w_h5, self.w_o]
        updates = nnet.RMSprop(self.cost, params, lr=0.001)

        self.train = theano.function(inputs=[X, Y], outputs=self.cost, updates=updates, allow_input_downcast=True)
        self.predict_ = theano.function(inputs=[X], outputs=self.py_x, allow_input_downcast=True)

    def fit(self, trX, trY):
        """
        Neural Network learning.
        :param trX: Input data for training (train X)
        :param trY: Output data for training (train y)

        """
        for i in range(100):
            shuffle = np.random.permutation(len(trY))
            trYs = trY[shuffle]
            trXs = trX[shuffle]
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
                cost = self.train(trXs[start:end], trYs[start:end])

    def predict(self, teX):
        """
        Neural network predicting.
        :param teX: Input data for predicting (test X)
        :return: Predictions
        """
        prY = self.predict_(teX)

        """ Randomize weights after training and predicting for new round """
        self.w_h.set_value(nnet.rand_weights((self.n_kernels, 1, 1, self.k * 4)))
        self.w_h2.set_value(nnet.rand_weights((self.n_kernels * 2, self.n_kernels, 1, self.k)))
        self.w_h3.set_value(nnet.rand_weights((self.n_kernels * 4, self.n_kernels * 2, 1, self.k)))
        self.w_h4.set_value(nnet.rand_weights((self.final_image_size, self.n_hidden_neurons)))
        self.w_h5.set_value(nnet.rand_weights((self.n_hidden_neurons, self.n_hidden_neurons)))
        self.w_o.set_value(nnet.rand_weights((self.n_hidden_neurons, self.class_size)))

        return prY
