import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
import tensorflow as tf

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

text = ["I like playing football with my friends"]


def tokenize(corpus):
    '''
    The Tokenizer stores everything in the word_index during fit_on_texts.
    Then, when calling the texts_to_sequences method, only the top num_words are considered.

    You can see that the value's are clearly not sorted after indexing.
    It is respected however in the texts_to_sequences method which turns input into numerical arrays:
    '''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    # check the index
    logging.info(tokenizer.word_index)

    corpusTokenized = tokenizer.texts_to_sequences(corpus)
    # check the tokens
    logging.info(corpusTokenized)

    V = len(tokenizer.word_index)
    return corpusTokenized, V


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def corpus2context(corpusTokenized, V, windowSize):
    '''
    get context and center hot vectors
    :param corpusTokenized:
    :param V:
    :param windowSize:
    :return:
    '''
    for words in corpusTokenized:
        L = len(words)
        for index, word in enumerate(words):
            contexts = []
            center = []
            start = index - windowSize
            end = index + windowSize + 1
            contexts.append([words[i] - 1 for i in range(start, end) if 0 <= i < L and i != index])
            center.append(word - 1)
            x = to_categorical(contexts, V)
            y = to_categorical(center, V)
            yield (x, y.ravel())


def softmax(x):
    '''
    Given an array of real numbers (including negative ones),
    the softmax function essentially returns a probability distribution with sum of the entries equal to one.
    :param x:
    :return:
    '''
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum(axis=0)


def skipgram(context, x, W1, W2, loss):
    h = np.dot(W1.T, x)
    u = np.dot(W2.T, h)
    y_pred = softmax(u)
    logging.info(y_pred)

    e = np.array([-label + y_pred.T for label in context])
    dW2 = np.outer(h, np.sum(e, axis=0))
    dW1 = np.outer(x, np.dot(W2, np.sum(e, axis=0)))
    new_W1 = W1 - eta * dW1
    new_W2 = W2 - eta * dW2
    loss += -np.sum([u[label == 1] for label in context]) + len(context) * np.log(np.sum(np.exp(u)))
    return new_W1, new_W2, loss


if __name__ == '__main__':
    corpusTokenized, V = tokenize(text)

    # initialize weights (with random values) and loss function
    N = 2  # assume that the hidden layer has dimensionality = 2
    window_size = 2  # symmetrical
    eta = 0.1  # learning rate
    np.random.seed(100)
    W1 = np.random.rand(V, N)
    W2 = np.random.rand(N, V)
    loss = 0.

    # for i, (x, y) in enumerate(corpus2context(corpusTokenized, V, 2)):
    #     print(i, "\n center word =", y, "\n context words =\n", x)
    #
    # data = [-1, -5, 1, 5, 3]
    # print(('softmax(data) = [' + 4 * '{:.4e}  ' + '{:.4e}]').format(*softmax(data)))
    # print('sum(softmax)  = {:.2f}'.format(np.sum(softmax(data))))

    for i, (label, center) in enumerate(corpus2context(corpusTokenized, V, 2)):
        W1, W2, loss = skipgram(label, center, W1, W2, loss)
        print("Training example #{} \n-------------------- \n\n \t label = {}, \n \t center = {}".format(i, label,
                                                                                                         center))
        print("\t W1 = {}\n\t W2 = {} \n\t loss = {}\n".format(W1, W2, loss))