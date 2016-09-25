import theano.tensor as T
import numpy as np
import theano

class OutputLayer(object):

    def __init__(self, rng, input, hidden_size, class_num, name, prefix=None, activation=T.tanh):

        W = np.array(
            rng.uniform(
                low = -np.sqrt(6.0 / (hidden_size + class_num)),
                high = np.sqrt(6.0 / (hidden_size + class_num)),
                size = (hidden_size, class_num)
            ),
            dtype=np.float32
        )

        b = np.zeros((class_num, ), dtype=np.float32)

        self.W = theano.shared(W, name='W', borrow=True)
        self.b = theano.shared(b, name='b', borrow=True)

        _output = T.dot(input, self.W) + self.b
        self.output = _output if activation == None else activation(_output)
        self.params = [self.W, self.b]

    def save(self):
        pass