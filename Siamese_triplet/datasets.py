import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


nb_classes=10
img_w, img_h=28,28
input_shape=(img_h, img_w, 1)
batch_size=32

def getDataset():
    (x_train_ori, y_train_ori), (x_test_ori, y_test_ori)=mnist.load_data()
    x_train_ori=x_train_ori.reshape(x_train_ori.shape[0],img_h,img_w, 1)
    x_test_ori=x_test_ori.reshape(x_test_ori.shape[0], img_h, img_w, 1)

    dataset_train=[]
    dataset_test=[]

    #sorting images by class id
    for n in range(nb_classes):
       images_class_n=np.asarray([row for idx, row in enumerate(x_train_ori) if y_train_ori[idx]==n])
       dataset_train.append(images_class_n/255)

       images_test_class_n=np.asarray([row for idx, row in enumerate(x_test_ori) if y_test_ori[idx]==n])
       dataset_test.append(images_test_class_n/255)

    return dataset_train, dataset_test,x_train_ori, y_train_ori, x_test_ori, y_test_ori

def get_random_triplet(dataset_train, dataset_test, size, s):
    #create batch of anchor, positive and negative by random selection within the classes

    if s=='train':
       X=dataset_train
    else:
       X=dataset_test
    _,w,h,c=X[0].shape
    #initilize
    #size, h, w, c=100,28,28,1
    triplets=[np.zeros((size,h,w,c)) for i in range(3)]

    for i in range(size):
        anchor_class=np.random.randint(0, nb_classes)
        nb_samples_class_AP=X[anchor_class].shape[0]
        [idx_A, idx_P]=np.random.choice(nb_samples_class_AP,size=2,replace=False)

        negative_class=(anchor_class+np.random.randint(1,nb_classes))%nb_classes
        nb_samples_class_N=X[negative_class].shape[0]
        idx_N=np.random.randint(0,nb_samples_class_N)

        triplets[0][i,:,:,:]=X[anchor_class][idx_A,:,:,:]
        triplets[1][i,:,:,:]=X[anchor_class][idx_P,:,:,:]
        triplets[2][i,:,:,:]=X[negative_class][idx_N,:,:,:]

    return triplets

def display_triplets(triplets_batch):

    b=triplets_batch[0].shape[0]
    labels=['Anchor','Positive', 'Negative']

    for i in range(batch_size):
        fig=plt.figure(figsize=(16,2))
        for j in range(3):
            subplot=fig.add_subplot(1,3,j+1)
            plt.imshow(triplets_batch[j][i,:,:,0], vmin=0,vmax=1,cmap='Greys')
            subplot.title.set_text(labels[j])
        plt.show()
        if i>5:
            break

    print ('ok')

#hard_size: number of hard triplets in a batch; random_size: number of random triplets in a batch
#hard_size+random_size=batch_size
def get_hard_triplet_batch(dataset_train, dataset_test,size, hard_size,random_size,network, s):

    if s=='train':
        X=dataset_train
    else:
        X=dataset_test

    _,w,h,c=X[0].shape

    random_batch=get_random_triplet(dataset_train, dataset_test,size,s)
    random_batch_loss=np.zeros((size)) #initialize loss for the random batch

    A=network.predict(random_batch[0])
    P=network.predict(random_batch[1])
    N=network.predict(random_batch[2])

    random_batch_loss=np.sum(np.square(A-P),axis=1)-np.sum(np.square(A-N),axis=1)
    #sort the loss by distance, the higher the harder, and select the hardest hard_size samples
    hard_select=np.argsort(random_batch_loss)[::-1][:hard_size]
    random_select=np.random.choice(np.delete(np.arange(size),hard_select),random_size, replace=False)
    selection=np.append(hard_select,random_select)

    triplets=[random_batch[0][selection,:,:,:], random_batch[1][selection,:,:,:], random_batch[2][selection,:,:,:]]

    return triplets






