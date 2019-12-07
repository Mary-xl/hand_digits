"""
========================
Build autoencoder model
========================
Author: Mary Li
Reference: https://blog.keras.io/building-autoencoders-in-keras.html
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D,MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard


def autoencoder_model(img_w,img_h,img_c):

    #encoding
    input_img=Input(shape=(img_w,img_h,img_c))
    x=Conv2D(16,(3,3), activation='relu',padding='same')(input_img)
    x=MaxPooling2D((2,2),padding='same')(x)
    x=Conv2D(8,(3,3),activation='relu',padding='same')(x)
    x=MaxPooling2D((2,2),padding='same')(x)
    x=Conv2D(8,(3,3),activation='relu',padding='same')(x)
    encoded=MaxPooling2D((2,2),padding='same', name='encoding_layer')(x)

    #decoding
    x=Conv2D(8,(3,3),activation='relu',padding='same')(encoded)
    x=UpSampling2D((2,2))(x)
    x=Conv2D(8,(3,3),activation='relu',padding='same')(x)
    x=UpSampling2D((2,2))(x)
    x=Conv2D(16,(3,3), activation='relu')(x)
    x=UpSampling2D((2,2))(x)
    decoded=Conv2D(1,(3,3), activation='sigmoid',padding='same')(x)

    model=Model(input_img,decoded)
    return model




