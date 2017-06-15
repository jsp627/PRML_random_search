import time
import numpy as np
import tensorflow as tf
from tf_multiclass_logistic import *
from load_mnist import load_mnist
import pandas as pd


mnist_path = 'Data\\MNIST'

mnist_tr = load_mnist(dataset='training', path = mnist_path)
mnist_test = load_mnist(dataset='testing', path = mnist_path)

summary = pd.DataFrame()

n_class = 10

start_time = time.time()
for i in range(200):
    # basis_type = ['sigmoid', 'rbf', 'tanh', 'identity'][np.random.choice(range(4))]
    basis_type = 'identity'
    n_basis = np.random.choice([2, 3, 4]) if basis_type != 'identity' else 1
    gamma = 10 ** np.random.uniform(-3, 0)

    shrink_order = np.random.choice([1, 2])
    lambda_weight = 10 ** np.random.uniform(-2, 1)
    lr = 10 ** np.random.uniform(-3, 0)
    n_feature = 784 * n_basis


    Phi = basis(mnist_tr[1], n_basis, 0, 255, gamma, basis_type) if basis_type != 'identity' else mnist_tr[1]

    model = LogisticRegression(n_feature, n_class, shrink_order, lambda_weight)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    for step in range(50):
        _, loss, acc = sess.run([model.sgd, model.cost, model.accuracy],
                                feed_dict = {model.X: Phi, model.Y: mnist_tr[0], model.learning_rate: lr})
        # if step % 20 == 0:
        print(step, loss, acc)
    Phi_test = basis(mnist_test[1], n_basis, 0, 255, gamma, basis_type) if basis_type != 'identity' else mnist_test[1]
    acc = sess.run(model.accuracy, feed_dict={model.X: Phi_test, model.Y: mnist_test[0]})

    sess.close()

    summary = summary.append(
        {'shrink': shrink_order,
         'lambda': lambda_weight,
         'basis_type': basis_type,
         'n_basis': n_basis,
         'learning': lr,
         # 'gamma': gamma,
         'accuracy': acc}, ignore_index=True)
end_time = time.time()
print(end_time-start_time)

summary.to_csv('mnist_logistic_random.csv')