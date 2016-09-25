import theano.tensor as T
import numpy as np
from theano import config
from optimizer import *
from EmbLayer import EmbLayer
from LSTMLayer import LSTMLayer
from PoolLayer import LastPoolLayer
from OutputLayer import OutputLayer



class LSTMModel(object):
    def __init__(self, learning_rate=0.001, optimizer=adadelta, embedding=None,
                 embedding_size=300, class_num=22, hidden_size=300, batch_size=32
                 ):

        self.lr = np.array(learning_rate, dtype=np.float32)

        y = T.ivector()
        x = T.imatrix()
        mask = T.fmatrix()
        lr = T.fscalar(name='lr')
        rng = np.random

        # build model
        layers = []

        layers.append(EmbLayer(rng, x, embedding, embedding_size, 'emblayer'))
        layers.append(LSTMLayer(rng, layers[-1].output, mask, embedding_size, hidden_size, 'wordlstmlayer'))
        layers.append(LastPoolLayer(layers[-1].output))
        layers.append(OutputLayer(rng, layers[-1].output, hidden_size, class_num, 'fulllayer', activation=T.nnet.softmax))

        self.layers = layers

        # get all params
        params = []
        for layer in layers:
            params += layer.params

        # define cost
        cost = -T.mean(T.log(layers[-1].output[T.arange(batch_size), y]), dtype=config.floatX)
        L2_rate = np.float32(1e-5)
        for param in params[1:]:
            cost += T.sum(L2_rate * (param * param), dtype=config.floatX)

        # get all gparams
        gparams = [T.grad(cost, param) for param in params]

        # define predict
        preds = T.argmax(layers[-1].output, axis=1)

        # define evaluation
        # precision = T.sum(T.eq(preds, y), dtype='int32')
        # err = preds - y
        # mean_squre_error = T.sum(err * err)
        # self.mean_squre_error = mean_squre_error

        # self.f_predict = theano.function([x, mask], [preds])
        #
        # tparams = dict(zip(map(str, range(0, len(params))), params))
        # for k in tparams:
        #     print(k, tparams[k])
        # self.f_cost, self.f_update = optimizer(lr, tparams, gparams, x, mask, y, cost)

        updates = AdaUpdates(params, gparams, 0.95, 1e-6)

        self.train_model = theano.function(
            inputs=[x, mask, y],
            outputs=cost,
            updates=updates,
        )

        self.test_model = theano.function(
            inputs=[x, mask],
            outputs=[preds],
        )

    def fit(self, x, mask, y):
        """

        :param x:
        :param mask:
        :param y:
        :return:
        """
        cost = self.train_model(x, mask, y)
        return cost

    def predict(self, x, mask):
        preds = self.test_model(x, mask)
        return preds
