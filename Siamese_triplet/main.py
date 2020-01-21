from Siamese_triplet.datasets import get_hard_triplet_batch
from Siamese_triplet.datasets import getDataset
from Siamese_triplet.model import build_basemodel
from Siamese_triplet.model import build_model

nb_classes=10
img_w, img_h=28,28
input_shape=(img_h, img_w, 1)
batch_size=32


if __name__=='__main__':


    dataset_train, dataset_test,x_train_ori, y_train_ori, x_test_ori, y_test_ori=get_hard_triplet_batch(dataset_train, dataset_test, hard_size, random_size, network, s='train')
