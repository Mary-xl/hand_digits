"""
=========================================
Recognizing hand-written digits using CNN
=========================================
Author: Mary Li
"""
import  keras
from keras.datasets import mnist
from keras.models import  Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='Same',activation='relu',input_shape=(28,28,1)))
    # model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
    #model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

if __name__=='__main__':

#prepare the data for processing====================================================
  batch_size=128
  num_classes=10
  epochs=10
  img_rows, img_cols=28, 28
  #x_train.shape=(60000,28,28)
  (x_train, y_train),(x_test, y_test)=mnist.load_data()

  if K.image_data_format()=='channels_first':
      x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
      x_test=x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
  else:
      x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
      x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

  #Normalization
  x_train=x_train/255.0
  x_test=x_test/255.0

  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  #convert to one-hot
  y_train=to_categorical(y_train,num_classes=num_classes)
  y_test=to_categorical(y_test,num_classes=num_classes)

  # data augmentation
  datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
  datagen.fit(x_train)
# data ready======================================================================

  model=build_model(num_classes)

  sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# training the model
#   history=model.fit_generator (datagen.flow(x_train, y_train,batch_size=batch_size),
#            epochs=epochs,
#            verbose=1,
#            validation_data=(x_test, y_test))
  history=model.fit(x_train, y_train,
          batch_size=batch_size,
          shuffle=True,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

  fig, ax = plt.subplots(2,1)
  ax[0].plot(history.history['loss'], color='b', label="Training loss")
  ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
  legend = ax[0].legend(loc='best', shadow=True)

  ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
  ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
  legend = ax[1].legend(loc='best', shadow=True)

  plt.show()
  print ("start test:=================================================================")
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])