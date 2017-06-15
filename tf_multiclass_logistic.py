import tensorflow as tf
import numpy as np

class LogisticRegression():
    def __init__(self, n_feature, n_class, lambda_weight, shrink_order):
        self.X = tf.placeholder(tf.float32, [None, n_feature])
        self.Y = tf.placeholder(tf.int32, [None])

        self.Y_onehot = tf.one_hot(self.Y, n_class, 1., 0., -1)

        self.W = tf.Variable(tf.random_uniform([n_feature, n_class], -1.0, 1.0))
        self.b = tf.Variable(tf.zeros([n_class]))

        self.logits = tf.matmul(self.X, self.W) + self.b
        # self.logits = tf.layers.dense(self.X, n_class)

        self.softmax = tf.nn.softmax(self.logits)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_onehot, logits=self.logits))

        self.regularization = lambda_weight*tf.norm(self.W, ord=shrink_order)**shrink_order

        self.cost = self.cross_entropy + self.regularization

        self.learning_rate = tf.placeholder(tf.float32)

        self.sgd = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.to_int64(self.Y))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def basis(X, n_basis, x_min, x_max, gamma, basis_type):
    mu = np.linspace(x_min, x_max, n_basis+2)[1:-1]
    if basis_type == 'rbf':
        result = np.hstack([np.exp(-gamma * (X - mu[i]) ** 2) for i in range(n_basis)])
    elif basis_type == 'sigmoid':
        result = np.hstack([1/(1+np.exp(-gamma*(X-mu[i]))) for i in range(n_basis)])
    elif basis_type == 'tanh':
        result = np.hstack([2 / (1 + np.exp(-gamma * (X - mu[i]))) - 1 for i in range(n_basis)])
    else:
        raise ValueError
    return result

