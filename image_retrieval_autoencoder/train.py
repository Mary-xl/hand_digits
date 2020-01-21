"""
========================
Train autoencoder model
========================
Author: Mary Li
Reference: https://blog.keras.io/building-autoencoders-in-keras.html
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def preprocess():
    (x_train, y_train),(x_test, y_test)=mnist.load_data()  #x_train.shape=(60000,28,28)
    x_train=x_train.astype(np.float32)/255.
    x_test=x_test.astype(np.float32)/255.
    x_train=np.reshape(x_train,(len(x_train),28,28,1)) #x_train.shape=(60000,28,28.1)
    x_test=np.reshape(x_test,(len(x_test),28,28,1))

    noise_factor=0.5
    x_train_noisy=x_train+noise_factor*np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy=x_test+noise_factor*np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return x_train_noisy,x_test_noisy,y_train,y_test,x_train,x_test

def train(model):

    x_train_noisy,x_test_noisy,y_train,y_test,x_train,x_test=preprocess()
    model.compile(optimizer='adadelta',loss='mse')
    model.fit(x_train_noisy,x_train,
              epochs=200,
              batch_size=128,
              shuffle=True,
              validation_data=(x_test_noisy,x_test),
              callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)]
              )
    model.save('../working/autoencoder.h5')