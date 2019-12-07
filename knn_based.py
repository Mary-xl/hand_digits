"""
=========================================
Recognizing hand-written digits using KNN
=========================================
Author: Mary Li
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, metrics

for n in range(1,20,2):
    digits=datasets.load_digits()
    images=digits.images
    n_samples=len(images)
    data=images.reshape((n_samples,-1))
    train_data=data[:n_samples//2]
    train_label=digits.target[:n_samples//2]
    test_data=data[n_samples//2:]
    test_label=digits.target[n_samples//2:]

    Kneigh=KNeighborsClassifier(n_neighbors=n)
    Kneigh.fit(train_data,train_label)
    test_predicts=Kneigh.predict(test_data)
    test_gt=digits.target[n_samples//2:]
    #metric_report=metrics.classification_report(test_gt,test_predicts)
    accuracy=metrics.accuracy_score(test_gt,test_predicts)
    #print (metrics.classification_report(test_gt,test_predicts))
    print (accuracy)

