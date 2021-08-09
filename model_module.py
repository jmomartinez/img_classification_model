from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing_module import preprocessing


class img_model():
    def __init__(self,epochs,input_shape,pool_size,kernel_size,path,data_choice,trans):
        self.epochs = epochs
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.path = path
        self.choice = data_choice
        self.trans = trans

        preprocessing_obj = preprocessing(self.choice,self.trans,self.path,(386, 354))
        train_iterator, val_iterator = preprocessing_obj.generator_init()
        self.train_it = train_iterator
        self.test_it = val_iterator

    def create_model(self):
        input_shape = self.input_shape
        p_size = self.pool_size
        k_size = self.kernel_size

        model = Sequential()
        model.add(Conv2D(filters=32,kernel_size = k_size,activation='relu',input_shape=input_shape))
        model.add(MaxPool2D(pool_size=p_size))

        model.add(Conv2D(filters=64,kernel_size = k_size,activation='relu'))
        model.add(MaxPool2D(pool_size=p_size))

        model.add(Conv2D(filters=128,kernel_size = k_size,activation='relu'))
        model.add(MaxPool2D(pool_size=p_size))

        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dense(10,activation='softmax'))

        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        return model

    # Future Implementation
    def tune_parameters(self):
        pass

    def train(self):
        model = self.create_model()
        # fit_model = model.fit(train_iterator,steps_per_epoch=len(train_iterator),epochs=self.epochs)
        fit_model = model.fit(self.train_it,steps_per_epoch=len(self.train_it),epochs=self.epochs)
        return model,fit_model

    def predict(self):
        model,fit_model = self.train()
        # _,accuracy = model.evaluate(val_iterator,steps=len(val_iterator))
        _,accuracy = model.evaluate(self.test_it,steps=len(self.test_it))

        print('Test Accuracy: {}%'.format(round(accuracy*100,4)))

        plt.plot(fit_model.history['loss'])
        plt.legend(['Train'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Epochs vs Loss')
        plt.show()