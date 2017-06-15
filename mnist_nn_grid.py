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
for shrink_order in [1]:
    for lambda_weight in [(10**np.linspace(-4, 0, 5))[4]]:
    # for lambda_weight in [0.01]:
        for n_hlayer in [2, 3, 4]:
    # n_hidden = [int(10**np.random.uniform(1.5, 3)) for i in range(n_hlayer)] # 레이어마다 다를 때
    #         for n_hidden1 in np.round(10**np.linspace(2, 3, 4)).astype(int):
            for n_hidden1 in np.round(10 ** np.linspace(2, 3, 4)).astype(int):
                for lr in 10**np.linspace(-3, 0, 4):
                    for decay in 10**np.linspace(-7, -4, 4):
                        # activation = ['sigmoid', 'relu', 'tanh'][np.random.choice(range(3))]
                        activation = 'tanh'
                        n_hidden = [n_hidden1 for i in range(n_hlayer)]


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
                    summary.to_csv('result/mnist_nn_grid4.csv')

end_time = time.time()

print(end_time-start_time)
