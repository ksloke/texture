# glcm training train-glcm-3.py
# double input
# max pooling for branch 1 removed - slight increase to 85% accuracy within 100 epochs
# last max pooling for branch 2 removed - slight increase to 84% accuracy within 100 epochs
# remove all max pooling for branch 2 - very poor 0.34 after 20 epochs while train acc is .95
# conv2D for branch 2 - 80% within 100 epochs


import tensorflow 
from tensorflow.keras import applications
from tensorflow.keras.layers import Input, concatenate, Dense, Dropout, Conv2D, Flatten, SeparableConv2D, MaxPool2D, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import utils
from tensorflow.keras import backend as Backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import Callback

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
import math


import matplotlib.image as matplotImage
import matplotlib.pyplot as plt
import numpy as np
import itertools

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
with open ('img.npy','rb') as f:
	images=np.load(f)
	labels=np.load(f)
print(images.shape)
print(labels.shape)

## Read from File -- change this
with open ('glcm-ii.npy','rb') as f:
	glcm=np.load(f)
	#labels=np.load(f)
print(glcm.shape)
#print(labels.shape)

le = LabelEncoder()
labels = le.fit_transform(labels)

#Model channel - change this
num_channels=21 #### 7 for original glcm.npy, 21 for glcm-ii.npy, 42 for glcm-2a.npy
h_input=41
w_input=41
num_epochs = 100
num_batch = 64
seed = 12
num_classes=68
# -------


tensorflow.random.set_seed(seed)
np.random.seed(seed)



###  MODEL - Branch 1
inputGLCM = Input(shape=(num_channels, h_input, w_input))
inputImg = Input(shape=(128,128,3))

x=SeparableConv2D(64, 3, activation='relu', data_format="channels_first")(inputGLCM)
#x=MaxPool2D(3) (x)

x=SeparableConv2D(64, 3, activation='relu')(x)
#x=MaxPool2D(2)(x)

x=SeparableConv2D(32, 3,activation='relu')(x)
#x=MaxPool2D(2)(x)
x=Flatten()(x)

###  MODEL - Branch 2
y=SeparableConv2D(64, 3, activation='relu', data_format="channels_last")(inputImg)
y=MaxPool2D(3) (y)

y=SeparableConv2D(64, 3, activation='relu')(y)
y=MaxPool2D(2)(y)

y=SeparableConv2D(32, 3,activation='relu')(y)
y=MaxPool2D(2)(y)
y=Flatten()(y)

## combine
combined=concatenate([x,y])
z=Dense(128, activation='relu')(combined)
z=Dense(num_classes, activation='softmax')(z)
##
model = Model(inputs=[inputGLCM, inputImg], outputs=z)

###
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


## Split the training and test set
x1_train, x1_test, y1_train, y1_test = train_test_split( glcm, labels, test_size=0.2, random_state=seed)
x2_train, x2_test, y2_train, y2_test = train_test_split( images, labels, test_size=0.2, random_state=seed)
print(x1_train.shape)
print(x1_test.shape)
print(y1_train.shape)
print(y1_test.shape)

print(x2_train.shape)
print(x2_test.shape)
print(y2_train.shape)
print(y2_test.shape)


y1_train = utils.to_categorical(y1_train, num_classes)
y1_test = utils.to_categorical(y1_test, num_classes)
print(y1_train.shape)
print(y1_test.shape)



# To see the models' architecture and layer names, run the following
model.summary()

## Normal Fit to model without augmentation on the input
history = model.fit([x1_train,x2_train], y1_train, validation_data=([x1_test,x2_test], y1_test), epochs= num_epochs, batch_size= num_batch, verbose=2, shuffle=True)

model.save('glcm-3')

# Final evaluation of the model
scores = model.evaluate([x1_test,x2_test], y1_test, verbose=1)
print("Error: %.2f%%" % (100-scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predict_classes=model.predict_classes([x1_test,x2_test])
#print(predict_classes)
#print(y_test)
true_classes = np.argmax(y1_test,1)
#print(true_classes)
np.set_printoptions(threshold=np.inf) # print all in array
cfm=confusion_matrix(true_classes,predict_classes)
#print(cfm)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cfm, classes=list(range(0,67)),
                      title='Confusion matrix, without normalization')

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