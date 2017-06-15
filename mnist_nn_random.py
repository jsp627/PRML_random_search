from keras.utils import to_categorical
import numpy as np
import time
from tf_nn import NeuralNetwork
from load_mnist import *
import pandas as pd




mnist_path = 'Data\\MNIST'

mnist_tr = load_mnist(dataset='training', path = mnist_path)
mnist_test = load_mnist(dataset='testing', path = mnist_path)

n_feature = 784
n_class = 10
# shrink_order = 2
# lambda_weight = .01
# n_hlayer = 2
# n_hidden = [100, 100]
# lr = 0.001
# activation = 'sigmoid'

# summary = pd.DataFrame(columns=['shrink', 'lambda', 'n_layer', 'n_hidden', 'learning rate', 'decay', 'activation'])

summary = pd.DataFrame()

start_time = time.time()
for i in range(20):
    shrink_order = np.random.choice([1])
    lambda_weight = 10**np.random.uniform(-4, 0)
    n_hlayer = np.random.choice([2, 3, 4])
    # n_hidden = [int(10**np.random.uniform(1.5, 3)) for i in range(n_hlayer)] # 레이어마다 다를 때
    n_hidden1 = int(10**np.random.uniform(2, 3))
    n_hidden = [n_hidden1 for i in range(n_hlayer)]
    lr = 10**np.random.uniform(-3, 0)
    decay = 10**np.random.uniform(-7, -4)
    # activation = ['sigmoid', 'relu', 'tanh'][np.random.choice(range(3))]
    activation = 'tanh'


    nn = NeuralNetwork(n_feature, n_class, lambda_weight, shrink_order, n_hlayer, n_hidden, activation, lr, decay)
 
    nn.fit(mnist_tr[1], to_categorical(mnist_tr[0], 10), batch_size=32, epochs=5)

    pred = nn.model.predict(mnist_test[1])

    acc = np.sum(pred.argmax(1) == mnist_test[0]) / mnist_test[0].shape[0]
    summary = summary.append(
        {'shrink': shrink_order,
         'lambda': lambda_weight,
         'n_layer': n_hlayer,
         'n_hidden': n_hidden1,
         'learning': lr,
         'decay': decay,
         'activation': activation,
         'accuracy': acc}, ignore_index=True)
    # summary.to_csv('result/mnist_nn_random5.csv')

end_time = time.time()

print(end_time-start_time)
