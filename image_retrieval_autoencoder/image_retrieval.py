"""
=====================================
Using the encoder for image retrieval
=====================================
Author: Mary Li
Reference:https://blog.keras.io/building-autoencoders-in-keras.html
"""

import numpy as np
from sklearn.neighbors import BallTree
from sklearn import preprocessing
from tensorflow.keras.models import load_model, Model
from image_retrieval_autoencoder.train import preprocess
from sklearn import preprocessing


def image_retrieval():
    topK=10
    avg_acc=0

    x_train_noisy, x_test_noisy, y_train, y_test, x_train, x_test=preprocess()
    autoencoder=load_model('../working/autoencoder.h5')
    print (autoencoder.summary())
    encoder=Model(autoencoder.input, autoencoder.get_layer('encoding_layer').output)

    coded_train=encoder.predict(x_train_noisy)
    coded_train=coded_train.reshape(coded_train.shape[0],coded_train.shape[1]*coded_train.shape[2]*coded_train.shape[3])
    coded_train = preprocessing.normalize(coded_train, norm='l2')

    tree=BallTree(coded_train,leaf_size=200)

    #extracting features from test set
    coded_test=encoder.predict(x_test_noisy)
    coded_test = coded_test.reshape(coded_test.shape[0], coded_test.shape[1] * coded_test.shape[2] * coded_test.shape[3])
    coded_test = preprocessing.normalize(coded_test, norm='l2')

    for i in range(coded_test.shape[0]):
        query_code=coded_test[i]
        query_label=y_test[i]
        dists,ids=tree.query([query_code],k=topK)
        labels=np.array([y_train[id] for id in ids[0]])

        acc=(labels==query_label).astype(int).sum()/topK
        avg_acc+=acc
        if i % 1000 == 0:
            print('{} / {}: {}'.format(i, coded_test.shape[0], acc))
    avg_acc/=coded_test.shape[0]
    print ("The average top K accuracy is: {}".format(avg_acc))
