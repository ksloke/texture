# name: train_00030_glcm_i21_separable.py
# Added EarlyStopping 12/9/2021 
# TO edit: Change npy file to open and number of classes

#00030 - separate test files each with different rotation. Train on 1360 samples, validate on 340 sample
#train 0 - 
# rot 1 : 0.9088 epoch 292
# rot 2 : 0.87058824 Epoch: 161
# ROT 3 : 0.87352943 Epoch: 277
# ROT 4 :  0.86764705 Epoch: 226
# ROT 5 : 0.85882354 Epoch: 223
# rot 6 :  0.84411764 Epoch: 247
# rot 7 : 0.85588235 Epoch: 300
# rot 8 :  0.8264706 Epoch: 235
import tensorflow 
from tensorflow.keras import applications
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, SeparableConv2D, MaxPool2D, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import utils
from tensorflow.keras import backend as Backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import Callback, EarlyStopping

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
#from scipy.misc import imread, imresize
from sklearn.model_selection  import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from functools import reduce
import math
from io import BytesIO

import os
import PIL.Image as Image
import numpy as np
import math


import matplotlib.image as matplotImage
import matplotlib.pyplot as plt
import numpy as np
import itertools

class stopAtAccuracyValue(Callback): #custom callback
        def on_epoch_end(self, epoch, logs=None):
            #keys = list(logs.keys())
            #print("End epoch {} of training; got log keys: {}".format(epoch, keys))
            THR = .9 #Assign THR with the value at which you want to stop training.
            if logs.get('val_accuracy') >= THR:
                 self.model.stop_training = True

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

## Read from File -- change this
## Split the training and test set
## Training set
with open ('outex-00030-0-glcm-i21-16L.npy','rb') as f:
	x_train=np.load(f)
	y_train=np.load(f)
print(x_train.shape)
print(y_train.shape)
## Test set
with open ('outex-00030-8-glcm-i21-16L.npy','rb') as f:
	test=np.load(f)
	test_labels=np.load(f)
print(test.shape)
print(test_labels.shape)

#we need to extract 5 from every class
x_test=np.asarray([test[i] for i in range(len(test)) if i%20 <5])
y_test=np.asarray([test_labels[i] for i in range(len(test_labels)) if i%20 <5])

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

#Model channel - change this
num_channels=24 #### , 24 for glcm-i21.npy
h_input=41
w_input=41
num_epochs = 300
num_batch = 64
seed = 12

num_classes=68 #outex
#num_classes=154 #mbt
# -------
tensorflow.random.set_seed(seed)
np.random.seed(seed)

y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

#print(x_train[0].shape)
#print(x_test[0].shape)
#print(y_train[0].shape)
#print(y_test[0])


###  MODEL
model = Sequential()
model.add(SeparableConv2D(64, (3, 3),input_shape=(num_channels, h_input, w_input), data_format="channels_first")) ##NOTE
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(SeparableConv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(SeparableConv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('sigmoid')) # relu
model.add(Dropout(0.3))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# To see the models' architecture and layer names, run the following
model.summary()

###
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
## Normal Fit to model without augmentation on the input
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs= num_epochs, 
                batch_size= num_batch, verbose=2, shuffle=False, callbacks=[stopAtAccuracyValue()])



# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=1)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print('Max Accuracy:', np.max(history.history['val_accuracy']))
print('Epoch:', np.argmax(history.history['val_accuracy'])+1)

#predict_classes=model.predict_classes(x_test) 
predict_classes=np.argmax(model.predict(x_test), axis=-1) #TF2.3
#print(predict_classes)
#print(y_test)
true_classes = np.argmax(y_test,1)
#print(true_classes)
np.set_printoptions(threshold=np.inf) # print all in array
cfm=confusion_matrix(true_classes,predict_classes)
#print(cfm)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cfm, classes=list(range(0,67)),
                      title='Confusion matrix')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cfm, classes=list(range(0,67)), normalize=True,title='Normalized')

plt.show()

#Graph
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()