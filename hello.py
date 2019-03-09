# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 13:00:18 2018

@author: Manoochehr
"""
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import  Conv1D, MaxPooling1D, Flatten, Dense, Activation, Dropout
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.metrics import log_loss
#from SignatureCroping import getSignatureFromPage

filenames = glob.glob("DatasetSig1/*/*.png")
imagesX = [cv2.imread(img,cv2.IMREAD_GRAYSCALE) for img in filenames]
y_all = []
for file in filenames:
        _,imgY,_ = file.split("\\")
        y_all.append(imgY)

X_all = []
for img in imagesX:              
        _,thresh1 = cv2.threshold(img,220,1,cv2.THRESH_BINARY_INV)         
        img = thresh1 * img        
        img = cv2.resize(img,(220, 150))                
        X_all.append(img)



y_all = np.array(y_all)
X_all = np.asarray(X_all)

plt.imshow(X_all[0,:,:])
#plt.imshow(X_all[35])

# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)


X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=0.1, random_state=23, 
                                                    stratify=y_all)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'
def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)
print('1')

classifier = Sequential()
#classifier.add(Activation( activation=center_normalize, input_shape=(220, 150, 1)))
classifier.add(Conv1D(96, 11, strides=4, padding= 'valid',input_shape=(150, 220)))
classifier.add(BatchNormalization())
classifier.add(Activation( activation='relu'))

classifier.add(MaxPooling1D(3, strides=2, padding='same'))

classifier.add(Conv1D(256, 5, strides=1, padding= 'same'))
classifier.add(BatchNormalization())
classifier.add(Activation( activation='relu'))

classifier.add(MaxPooling1D(3, strides=2, padding='same'))

classifier.add(Conv1D(384, 3, strides=1, padding= 'same'))
classifier.add(BatchNormalization())
classifier.add(Activation( activation='relu'))

classifier.add(Conv1D(384, 3, strides=1, padding= 'same'))
classifier.add(BatchNormalization())
classifier.add(Activation( activation='relu'))

classifier.add(Conv1D(256, 3, strides=1, padding= 'same'))
classifier.add(BatchNormalization())
classifier.add(Activation(activation='relu'))

classifier.add(MaxPooling1D(3, strides=2, padding='same'))

classifier.add(Flatten())
classifier.add(Dense(units = 2048))
classifier.add(BatchNormalization())
classifier.add(Activation(activation='relu'))
classifier.add(Dense(units = 2048))
classifier.add(BatchNormalization())
classifier.add(Activation( activation='relu'))

classifier.add(Dense(units = 10 , activation='softmax')) # 10 is number of targets.

#classifier.add(Dense(units = 10, activation = 'sigmoid'))

adam = Adam(lr=0.001)

# For a mean squared error regression problem
# For a multi-class classification problem
sgd = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True, clipnorm=1.)
classifier.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

classifier.fit(X_train, y_train,
          epochs=10,
          batch_size=32)
score = classifier.evaluate(X_valid, y_valid, batch_size=32)
preds = classifier.predict(X_valid, verbose=1)
#
#
# with a Sequential model
get_nth_layer_output = K.function([classifier.layers[0].input],
                                  [classifier.layers[-2].output])
FeatureLayer = get_nth_layer_output([X_train])[0]

FeatureLayer = np.asarray(FeatureLayer)
a = np.histogram(FeatureLayer)

plt.hist(a, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()






#from sklearn.svm import SVC
#
#clf = SVC(kernel = 'linear',random_state=0,gamma='scale', decision_function_shape='ovo')
#clf.fit(FeatureLayer, y_valid) 







