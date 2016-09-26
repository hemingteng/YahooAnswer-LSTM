import re, os
from gensim.models import word2vec
import numpy as np
import cPickle as pickle


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def load_data(data_dir):
    """
    load_data
    :param data_dir:
    :return:
        train_data: ([word1, word2, word3, \ldot, wordn], lable)
        valid_data: ([word1, word2, word3, \ldot, wordn], lable)
        test_data:  ([word1, word2, word3, \ldot, wordn], lable)
    """
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    train_file = [f for f in files if 'train' in f][0]
    valid_file = [f for f in files if 'dev' in f][0]
    test_file = [f for f in files if 'test' in f][0]

    def get_data(file):
        with open(file, 'r') as f:
            data = []
            for line in f:
                line = str.lower(line.strip())
                label, sentence = line.split('\t')
                sentence = tokenize(sentence)
                data.append((sentence, int(label)))
            return data

    train_data = get_data(train_file)
    valid_data = get_data(valid_file)
    test_data = get_data(test_file)
    return train_data, valid_data, test_data


def load_embedding(data, embedding_file, binary=True, prefix=None, file_name='embedding.pkl'):
    """

    :param data:
    :param embedding_file:
    :param binary:
    :param prefix: if prefix is None, then write to file_name, else load from prefix
    :param file_name:
    :return:
    """
    if prefix == None:
        vocab = sorted(reduce(lambda x, y: x | y, (set(sentence) for sentence, _ in data)))
        word_idx = dict((c, i) for i, c in enumerate(vocab))
        vocab_size = len(word_idx) + 1  # +1 for nil word

        # "/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin"
        model = word2vec.Word2Vec.load_word2vec_format(embedding_file, binary=binary)

        embedding = []
        for c in word_idx:
            if c in model:
                embedding.append(model[c])
            else:
                embedding.append(np.random.uniform(0.1, 0.1, 300))
        embedding = np.array(embedding, dtype=np.float32)
        with open(file_name, 'wb') as f:
            pickle.dump(embedding, f)
            pickle.dump(vocab_size, f)
            pickle.dump(word_idx, f)
    else:
        with open(prefix, 'rb') as f:
            embedding = pickle.load(f)
            vocab_size = pickle.load(f)
            word_idx = pickle.load(f)

    return vocab_size, word_idx, embedding


def vectorize_data(data, word_idx, sentence_size, class_num):
    sentences, masks, labels = [], [], []
    # masks = np.zeros((data_size, sentence_size))
    for sentence, label in data:
        # pad to memory_size
        length = len(sentence)
        l_s = max(0, sentence_size - len(sentence))
        sentence = [word_idx[w] for w in sentence] + [0] * l_s
        mask = [1.0] * length + [0.0] * l_s
        sentences.append(sentence)
        masks.append(mask)
        labels.append(label)

    masks = np.array(masks, dtype=np.float32)
    masks = np.transpose(masks, (1, 0))

    sentences = np.array(sentences, dtype=np.int32)
    sentences = np.transpose(sentences, (1, 0))
    return np.array(sentences, dtype=np.int32), np.array(masks, dtype=np.float32), np.array(labels, dtype=np.int32)
