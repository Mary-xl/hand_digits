
import numpy as np
from tensorflow.keras.utils import plot_model,normalize
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import math
from Siamese_triplet.datasets import getDataset
from Siamese_triplet.model import build_basemodel
from Siamese_triplet.model import TripletLossLayer
nb_classes=10
img_w, img_h=28,28
input_shape=(img_h, img_w, 1)
embeddingsize=10


def compute_dist(a,b):
    return np.sum(np.square(a-b))
def compute_probs(network ,X ,Y):
    '''
    Input
        network : current NN to compute embeddings
        X : tensor of shape (m,w,h,1) containing pics to evaluate
        Y : tensor of shape (m,) containing true class

    Returns
        probs : array of shape (m,m) containing distances

    '''
    m = X.shape[0]
    nbevaluation = int( m *( m -1 ) /2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))

    # Compute all embeddings for all pics with current network
    embeddings = network.predict(X)

    size_embedding = embeddings.shape[1]

    # For each pics of our dataset
    k = 0
    for i in range(m):
        # Against all other images
        for j in range( i +1 ,m):
            # compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
            probs[k] = -compute_dist(embeddings[i ,:] ,embeddings[j ,:])
            if (Y[i ]==Y[j]):
                y[k] = 1
                # print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
            else:
                y[k] = 0
                # print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
            k += 1
    return probs ,y


# probs,yprobs = compute_probs(network,x_test_origin[:10,:,:,:],y_test_origin[:10])

def compute_metrics(probs ,yprobs):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(yprobs, probs)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(yprobs, probs)

    return fpr, tpr, thresholds ,auc

def compute_interdist(dataset_test,network):
    '''
    Computes sum of distances between all classes embeddings on our reference test image:
        d(0,1) + d(0,2) + ... + d(0,9) + d(1,2) + d(1,3) + ... d(8,9)
        A good model should have a large distance between all theses embeddings

    Returns:
        array of shape (nb_classes,nb_classes)
    '''
    res = np.zeros((nb_classes ,nb_classes))

    ref_images = np.zeros((nb_classes ,img_h ,img_w ,1))

    # generates embeddings for reference images
    for i in range(nb_classes):
        ref_images[i ,: ,: ,:] = dataset_test[i][0 ,: ,: ,:]
    ref_embeddings = network.predict(ref_images)

    for i in range(nb_classes):
        for j in range(nb_classes):
            res[i ,j] = compute_dist(ref_embeddings[i] ,ref_embeddings[j])
    return res

def draw_interdist(dataset_test, network ,n_iteration):

    interdist = compute_interdist(dataset_test,network)
    data = []
    for i in range(nb_classes):
        data.append(np.delete(interdist[i ,:] ,[i]))

    fig, ax = plt.subplots()
    ax.set_title('Evaluating embeddings distance from each other after {0} iterations'.format(n_iteration))
    ax.set_ylim([0 ,3])
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    ax.boxplot(data ,showfliers=False ,showbox=True)
    locs, labels = plt.xticks()
    plt.xticks(locs ,np.arange(nb_classes))

    plt.show()

def find_nearest(array ,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx -1]) < math.fabs(value - array[idx])):
        return array[idx -1] ,idx -1
    else:
        return array[idx] ,idx

def draw_roc(fpr, tpr ,auc, thresholds):
    # find threshold
    targetfpr =1e-3
    _, idx = find_nearest(fpr ,targetfpr)
    threshold = thresholds[idx]
    recall = tpr[idx]


    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC: {0:.3f}\nSensitivity : {2:.1%} @FPR={1:.0e}\nThreshold={3})'.format(auc,targetfpr ,recall
                                                                                        ,abs(threshold) ))
    # show the plot
    plt.show()

#evaluate on untrained feature extractor base network
def evalute_model(network,dataset_test,x_test_ori, y_test_ori ):
    probs,yprob=compute_probs(network, x_test_ori[:500,:,:,:], y_test_ori[:500])
    probs, yprob = compute_probs(network, x_test_ori[:500, :, :, :], y_test_ori[:500])
    fpr, tpr, thresholds, auc = compute_metrics(probs, yprob)
    draw_roc(fpr, tpr, auc,thresholds)
    draw_interdist(dataset_test, network, n_iteration=0)


if __name__=='__main__':
    dataset_train, dataset_test, x_train_ori, y_train_ori, x_test_ori, y_test_ori=getDataset()
    #base=build_basemodel(input_shape, embeddingsize)

    # evalute_model(base, dataset_test, x_test_ori,y_test_ori)

    with CustomObjectScope({'TripletLossLayer': TripletLossLayer}):
         trained_model = load_model('../working/mnist_triplet.h5')
         trained_model.summary()

    base_model=load_model('../working/mnist_base.h5')
    base_model.summary()
    evalute_model(base_model, dataset_test, x_test_ori, y_test_ori)
