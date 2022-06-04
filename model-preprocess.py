# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:52:28 2020

@author: Shabista
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import regularizers

from collections import Counter


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#import matplotlib.pyplot as plt


#import pydot


def removeHTMLtags(blob):
     soup = BeautifulSoup(blob,  "html.parser").text
     return soup
    
def removeNonAlphaAndSpace(blob):
    words = list(blob.split(" "))  
    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = "space" 
    words[:]=[value for value in words if value != "space"]
    blob= ''.join(words)
    return blob


print("TensorFlow Version :",tf.__version__)

if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:
    print("Please install GPU version of TF")
    
    
train_data= pd.read_csv("train_data.csv")
print("train-csv-uploaded")
train_data.dropna(axis = 0, how ='any',inplace=True)
train_data['Num_words_text'] = train_data['message'].apply(lambda x:len(str(x).split())) 
mask = train_data['Num_words_text'] >1
train_data = train_data[mask]
print('===========Train Data =========')
print(train_data['class'].value_counts())
print(len(train_data))
print('==============================')

train_data['message'] = train_data['message'].apply(removeHTMLtags)
train_data['message'] = train_data['message'].apply(removeNonAlphaAndSpace)




test_data= pd.read_csv("test_data.csv" ,encoding='latin1')
print("test-csv-uploaded")
test_data.dropna(axis = 0, how ='any',inplace=True)
test_data['Num_words_text'] = test_data['message'].apply(lambda x:len(str(x).split())) 
mask = test_data['Num_words_text'] >1
test_data = test_data[mask]
print('===========Test Data =========')
print(test_data['class'].value_counts())
print(len(test_data))
print('==============================')

test_data['message'] = test_data['message'].apply(removeHTMLtags)
test_data['message'] = train_data['message'].apply(removeNonAlphaAndSpace)

X_train, X_test, y_train, y_test =train_test_split(train_data['message'].tolist(), train_data['class'].tolist(), test_size=0.33,stratify = train_data['class'].tolist(), random_state=0)

print('Train data len:'+str(len(X_train)))
print('Class distribution: '+str(Counter(y_train)))
print('Valid data len:'+str(len(X_test)))
print('Class distribution: '+ str(Counter(y_test)))

x_train=np.asarray(X_train)
x_valid = np.array(X_test)
x_test =np.asarray(test_data['message'].tolist())

le = LabelEncoder()

train_labels = le.fit_transform(y_train)
train_labels = np.asarray( tf.keras.utils.to_categorical(train_labels))

valid_labels = le.transform(y_test)
valid_labels = np.asarray( tf.keras.utils.to_categorical(valid_labels))

test_labels = le.transform(test_data['class'].tolist())
test_labels = np.asarray(tf.keras.utils.to_categorical(test_labels))
list(le.classes_)


train_ds = tf.data.Dataset.from_tensor_slices((x_train,train_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid,valid_labels))
test_ds = tf.data.Dataset.from_tensor_slices((x_test,test_labels))

print(y_train[:10])
train_labels = le.fit_transform(y_train)
print('Text to number')
print(train_labels[:10])
train_labels = np.asarray( tf.keras.utils.to_categorical(train_labels))
print('Number to category')
print(train_labels[:10])


count =0
print('======Train dataset ====')
for value,label in train_ds:
    count += 1
    #print(value,label)
    if count==5:
        break
count =0
print('======Validation dataset ====')
for value,label in valid_ds:
    count += 1
    #print(value,label)
    if count == 5:
        break
print('======Test dataset ====')
for value,label in test_ds:
    count += 1
    # print(value,label)
    if count==5:
        break
    
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
print(x_train[:1])
hub_layer(x_train[:1])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(1, activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))



model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=["CategoricalAccuracy"])


epochs = 1

# Fit the model using the train and test datasets.
history = model.fit(x_train, train_labels,validation_data= (x_test,test_labels),epochs=epochs )

print(history.history)

plt.plot(history.history['loss'], label='training data')
plt.plot(history.history['val_loss'], label='validation data')
plt.title('Loss for Text Classification')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()