import theano

class EmbLayer(object):
    def __init__(self, rng, input, embedding, embdding_size, name, prefix=None):
        self.name = name
        W = embedding
        self.W = theano.shared(W, name='E', borrow=True)
        self.output = self.W[input.flatten()].reshape((input.shape[0], input.shape[1], embdding_size))
        self.params = [self.W]

    def save(self, prefix):
        pass