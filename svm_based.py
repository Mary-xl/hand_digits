"""
=========================================
Recognizing hand-written digits using SVM
=========================================

This is an example from scikit-learn tutorial showing how to recognize images of
hand-written digits using SVM

"""

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt

digits = datasets.load_digits()
data_and_label=list(zip(digits.images, digits.target))
for idx, (image, label) in enumerate(data_and_label[25:33]):
    plt.subplot(2,4, idx+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample: %i'%label)

plt.show()
n_samples=len(digits.images)
images=digits.images
data=digits.images.reshape((n_samples,-1))
classifier=svm.SVC(gamma=0.001)
classifier.fit(data[:n_samples//2],digits.target[:n_samples//2])
predictions=classifier.predict(data[n_samples//2:])
gt=digits.target[n_samples//2:]
result=list(zip(predictions,gt))
print (metrics.classification_report(gt,predictions))


# clf=svm.SVC(gamma=0.001, C=100.)
# clf.fit(digits.data[:-1],digits.target[:-1])
# clf.predict(digits.data[-1])

print ('ok')