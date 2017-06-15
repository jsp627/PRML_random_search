from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from keras.optimizers import SGD



class NeuralNetwork():
    def __init__(self, n_feature, n_class, lambda_weight, shrink_order, n_hlayer, n_hidden, act, lr, decay):
        if shrink_order == 1:
            regularizer = regularizers.l1(lambda_weight)
        else:
            regularizer = regularizers.l2(lambda_weight)
        self.model = Sequential()
        self.model.add(Dense(n_hidden[0], input_shape=(n_feature, ), kernel_initializer='glorot_normal', kernel_regularizer=regularizer))
        for i in range(n_hlayer-1):
            self.model.add(Activation(act))
            self.model.add(Dense(n_hidden[i+1], kernel_initializer='glorot_normal', kernel_regularizer=regularizer))
        self.model.add(Activation(act))
        self.model.add(Dense(n_class, kernel_initializer='glorot_normal', kernel_regularizer=regularizer))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr, decay=decay),
                      metric=['accuracy'])
        self.fit = self.model.fit
