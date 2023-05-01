#results 75-80 (max) 0.7426 100 epoch, 0.7721 104 epoch, 0.7353 120 epoch
#with open ('img.npy','rb') as f:
#	images=np.load(f)
#	labels=np.load(f)
#print(images.shape)
#print(labels.shape)

# separable conv2d with glcm-i.npy (single color pair and reduced image 41x41)


import tensorflow 
from tensorflow.keras import applications
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, SeparableConv2D, MaxPool2D, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras import utils
from tensorflow.keras import backend as Backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import Callback

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

le = LabelEncoder()
labels = le.fit_transform(labels)

#Model channel - change this
num_channels=3 #### 7 for original glcm.npy,10 for glcm-i.npy, 21 for glcm-ii.npy, 42 for glcm-2a.npy
h_input=128
w_input=128
num_epochs = 120
num_batch = 64
seed = 12
num_classes=68
# -------


tensorflow.random.set_seed(seed)
np.random.seed(seed)

## Split the training and test set
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=seed)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

###  MODEL
model = Sequential()
model.add(SeparableConv2D(64, (3, 3),input_shape=(h_input, w_input,num_channels), data_format="channels_last")) ##NOTE
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
#print(new_model.summary())

###
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

### ONLY USE EITHER Model.fit or Model.fit_generator. DO NOT RUN BOTH!

## Normal Fit to model without augmentation on the input
history = model.fit(x_train, y_train, validation_data=[x_test, y_test], epochs= num_epochs, batch_size= num_batch, verbose=2, shuffle=True)

## Fit to model with augmentation on the input data. Note that steps_per_epoch should be the data length/batch_size.
#datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
#datagen.fit(x_train)
#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=num_batch), validation_data=(x_test, y_test), steps_per_epoch=len(x_train) / num_batch, epochs=num_epochs, verbose=2, shuffle=True)


#model.save('glcm-haralick')

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=1)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

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