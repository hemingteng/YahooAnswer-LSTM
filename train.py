from __future__ import absolute_import
from __future__ import print_function

from sklearn import metrics
from model.LSTMModel import LSTMModel
from process import *
import pyprind
import lasagne

hidden_size = 150
class_num = 22
max_epochs = 100
batch_size = 32
evaluation_interval = 1
embedding_file = "/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin"

print("Loading Task")

train_data, valid_data, test_data = load_data('./data')
data = train_data + valid_data + test_data

print("Loading Embedding")

vocab_size, word_idx, embedding = load_embedding(data, embedding_file, prefix='embedding.pkl') # load_embedding(data, prefix='embedding.pkl')  # load_embedding(data)

sentence_size = max(map(len, [sentence for sentence, label in data]))

print("Embedding Size", embedding.shape)
print("Sentence Size", sentence_size)

# train_sents is transpose matrix of origin, which means the dimension is (sent_size, n_train)
train_sents, train_masks, train_labels = vectorize_data(train_data, word_idx, sentence_size, class_num)
valid_sents, valid_masks, valid_labels = vectorize_data(valid_data, word_idx, sentence_size, class_num)
test_sents, test_masks, test_labels = vectorize_data(test_data, word_idx, sentence_size, class_num)

n_train = len(train_sents[0])
n_valid = len(valid_sents[0])
n_test = len(test_sents[0])

print("train data", n_train)
print("valid data", n_valid)
print("test data", n_test)


model = LSTMModel(hidden_size=hidden_size, batch_size=batch_size,  embedding=embedding)


batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
for t in range(max_epochs):
    total_cost = 0.0
    preds = []
    labels = []
    np.random.shuffle(train_data)

    process_bar = pyprind.ProgPercent(n_train / batch_size)
    for start in range(0, n_train, batch_size):
        process_bar.update()
        end = start + batch_size
        sent = train_sents[:,start:end]
        mask = train_masks[:,start:end]
        label = train_labels[start:end]

        cost_t = model.fit(sent, mask, label)
        pred = model.predict(sent, mask)

        total_cost += cost_t
        preds += list(pred)
        labels += list(label)

        #print("cost_t", cost_t, "epochs", start)

    train_acc = metrics.accuracy_score(labels, preds)
    print('Training Accuracy:', train_acc)

    if t % evaluation_interval == 0:

        val_preds = model.predict(valid_sents, valid_masks)
        val_acc = metrics.accuracy_score(list(valid_labels), list(val_preds))

        print('-----------------------')
        print('Epoch', t)
        print('Total Cost:', total_cost * batch_size / n_train)
        print('Training Accuracy:', train_acc)
        print('Validation Accuracy:', val_acc)
        print('-----------------------')

        test_preds = model.predict(list(test_sents), list(test_masks))
        test_acc = metrics.accuracy_score(test_labels, test_preds)
        print("Testing Accuracy:", test_acc)