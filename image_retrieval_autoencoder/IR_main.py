"""
==================================================================================
Using autoencoder to encode hand-written digits images for content based retrieval
==================================================================================
Author: Mary Li
Reference: https://blog.keras.io/building-autoencoders-in-keras.html
"""

from image_retrieval_autoencoder.train import train
from image_retrieval_autoencoder.autoencoder import autoencoder_model
from image_retrieval_autoencoder.test import test_encoder
from image_retrieval_autoencoder.image_retrieval import image_retrieval

if __name__=='__main__':

    autoencoder=autoencoder_model(28,28,1)
    # train(autoencoder)
    # test_encoder()
    # image_retrieval()

