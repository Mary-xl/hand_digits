"""
========================
Test the encoder-decoder
========================
Author: Mary Li
Reference: https://blog.keras.io/building-autoencoders-in-keras.html
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from image_retrieval_autoencoder.train import preprocess
import cv2

import matplotlib.pyplot as plt


def test_encoder():
    autoencoder=load_model('../working/autoencoder.h5')
    x_train_noisy, x_test_noisy, y_train, y_test, x_train, x_test = preprocess()
    decoded_imgs=autoencoder.predict(x_test_noisy)
    decoded_imgs*=255
    # for i in range(0,10):
    #     input_img=cv2.resize(x_test[i],(280,280))*255
    #     noisy_img=cv2.resize(x_test_noisy[i],(280,280))*255
    #     output_img=cv2.resize(denoised_test[i].reshape(28, 28),(280,280))*255
    #     cv2.imwrite('../result/'+str(i)+'_input.png',input_img)
    #     cv2.imwrite('../result/' + str(i) + '_noisy.png', noisy_img)
    #     cv2.imwrite('../result/' + str(i) + '_output.png', output_img)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        #display the noisy input
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    print ('ok')