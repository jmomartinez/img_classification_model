import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class preprocessing():
    def __init__(self,data_choice,transformation,path,input_shape):
        self.data_choice = data_choice
        self.transformation = transformation
        self.path = path
        self.input_shape = input_shape

    def datagen_flow(self,data):
        (x_train,y_train),(x_test,y_test) = data
        train_shape = x_train.shape
        test_shape = x_test.shape

        x_train = x_train.reshape(train_shape[0],train_shape[1],train_shape[2],1)
        x_test = x_test.reshape(test_shape[0],test_shape[1],test_shape[2],1)

        if self.transformation == 'scale':
            featurewise = True
            featurewise_std = True
            datagen = ImageDataGenerator(featurewise_center=featurewise,
            featurewise_std_normalization=featurewise_std)
            
            datagen.fit(x_train)
            train_iterator = datagen.flow(x_train,y_train,batch_size=64)
            val_iterator = datagen.flow(x_test,y_test,batch_size=64)
            return train_iterator,val_iterator

        elif self.transformation == 'normalize' and self.data_choice != 'path':
            # f = (x-x_min)/(x_max-x_min) = (x-0)/(255-0) = x/255
            scale_factor = float((1-x_train.min())/(x_train.max()-x_train.min()))
            datagen = ImageDataGenerator(rescale=scale_factor)

            train_iterator = datagen.flow(x_train,y_train,batch_size=64)
            val_iterator = datagen.flow(x_test,y_test,batch_size=64)
            return train_iterator,val_iterator

    def datagen_flow_from_dir(self):
        if self.transformation == 'normalize':
            # f = (x-x_min)/(x_max-x_min) = (x-0)/(255-0) = x/255
            # scale_factor = float((1-x_train.min())/(x_train.max()-x_train.min()))
            datagen = ImageDataGenerator(rescale = 1.0/255.0, validation_split=.15)

            train_iterator = datagen.flow_from_directory(self.path,batch_size=15,
            target_size = self.input_shape,color_mode='grayscale',subset='training',class_mode='binary')
            
            val_iterator = datagen.flow_from_directory(self.path,batch_size=5,
            target_size = self.input_shape,color_mode='grayscale',subset='validation',class_mode='binary')
            return train_iterator,val_iterator

    def generator_init(self):
        choice = self.data_choice.lower()

        if choice == 'fashion_mnist':
            data = fashion_mnist.load_data()
            train_iterator,val_iterator = self.datagen_flow(data)
            return train_iterator,val_iterator

        elif choice == 'mnist':
            data = mnist.load_data()
            train_iterator,val_iterator = self.datagen_flow(data)
            return train_iterator,val_iterator

        elif choice == 'path':
            train_iterator,val_iterator = self.datagen_flow_from_dir()
            return train_iterator,val_iterator