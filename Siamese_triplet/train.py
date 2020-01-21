import numpy as np
from tensorflow.keras.optimizers import Adam
from Siamese_triplet.datasets import get_hard_triplet_batch
from Siamese_triplet.model import build_model
from Siamese_triplet.model import build_basemodel
from Siamese_triplet.datasets import getDataset
from Siamese_triplet.evaluation import compute_dist, compute_probs
from tensorflow.keras.utils import plot_model,normalize
from sklearn.metrics import roc_curve,roc_auc_score

nb_classes=10
img_w, img_h=28,28
input_shape=(img_h, img_w, 1)
batch_size=32
size=200
evaluate_every = 1000 # interval for evaluating on one-shot tasks
num_iter = 20000 # No. of training iterations
n_val = 250


def train_model(num_iter):
    dataset_train, dataset_test, x_train_ori, y_train_ori, x_test_ori, y_test_ori=getDataset()
    base=build_basemodel(input_shape, embeddingsize=10)
    model=build_model(input_shape,base,margin=0.2)
    optimizer=Adam(lr=0.00006)
    model.compile(optimizer,None)
    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='../working/model.png')

    print ("start training.................")

    for i in range (1,num_iter+1):
        triplets_batch=get_hard_triplet_batch(dataset_train, dataset_test, size, int(batch_size*0.5), int(batch_size*0.5),base,s='train')
        loss=model.train_on_batch(triplets_batch, None)

        if i % evaluate_every == 0:
            print("\n ------------- \n")
            print("{0} iterations: , Train Loss: {1}".format(i,loss))
            probs, yprob = compute_probs(base, x_test_ori[:n_val, :, :, :], y_test_ori[:n_val])

    model.save('../working/mnist_triplet.h5')
    base.save('../working/mnist_base.h5')

if __name__=='__main__':
    train_model(num_iter)

