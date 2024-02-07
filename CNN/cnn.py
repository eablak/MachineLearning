# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:23:04 2024

@author: esraablak
"""

"""
Convolutional Neural Network
    - Resim
    - Convolutional and Non-Linear: Evrişim aslında bir filtredir ve bir dönüşüm operatörü olarak düşünülebilir. Yani gelen resme bir filtreler uygularsın. yeni hali (convolution matrisi) ile işlemlere devam ederesin. (
    - Pooling: Maks havuzlama / Ortalama havuzlama  => gelen matriste seçtiğin yönteme göre yeni daha küçük boyutta (önemli özniteliklerini koruyarak küçültme) matris oluşturursun.
    - Flatting: Poolingen gelen matrisin son halini düz bir şekile ve input neuron'lara dönüştürülmesi.
    
Bu adımları istediğin kadar yapabilirsin.
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
# Convolution
classifier.add(Convolution2D(32,3,3, input_shape= (64,64,3), activation="relu"))
# Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flatting
classifier.add(Flatten())

# Artificial Neural Network
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

# CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale= 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_dataframe("veriler/training_set", target_size=(64,64), batch_size=1, class_mode="binary")
test_set = test_datagen.flow_from_dataframe("veriler/test_set", target_size=(64,64), batch_size=1, class_mode="binary")

classifier.fit_generator(training_set, samples_per_epoch=8000, nb_epoch=1, validation_data = test_set, nb_val_samples= 2000)

import pandas as pd
import numpy as np

test_set.rest()
pred = classifier.predict_generator(test_set,verbose=1)
pred[pred> .5] = 1
pred[pred <= .5] = 0

test_labels = []

for i in range(0,int(203)):
    test_labels.extend(np.array(test_set[i][1]))

print("test_labels")
print(test_labels)

file_name = test_set.filenames

result = pd.DataFrame()
result["file_name"] = file_name
result["predicts"] = pred
result["test"] = test_labels

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels,pred)
print(cm)