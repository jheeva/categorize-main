# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:40:05 2022

@author: End User
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import  pickle
import numpy as np
import json
import datetime
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding,Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from tensorflow.keras.callbacks import TensorBoard




SCALER_SAVE_PATH = os.path.join(os.getcwd(), "saved_models", "category_scaler.pkl")
SCALER_SAVE_PATH1 = os.path.join(os.getcwd(), "saved_models", "text_scaler.pkl")
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'saved_models','model.h5')

PATH_LOGS=os.path.join(os.getcwd(),'accessment3','logs')
url='https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'




#%%step)1data loading
df=pd.read_csv(url)

category=df['category']
category_dummy=category.copy()

text=df['text']
text_dummy=text.copy()


#step 2)data inspection


category_dummy[3]
text_dummy[3]

category_dummy[10]
text_dummy[7]

df.tail()
df.isna().sum()


df.category.value_counts().plot(figsize=(12,5),kind='bar',color='green');
plt.xlabel('Category')
plt.ylabel('Total Number of Category for Training')

df.category.value_counts().plot(figsize=(12,5),kind='line',color='red');
plt.xlabel('Category')
plt.ylabel('Total Number Category for Training')


#step3)data cleaning


for index,text in enumerate(category_dummy):
    category_dummy[index]=re.sub('[a-zA-Z]',' ',text).lower().split()
    



#step 4)feature selection
#step 5)data preprocessing
#data vectorization

num_words=1000
oov_token='<OOV>'

#create tokenizer
tokenizer=Tokenizer(num_words=num_words,oov_token=oov_token)
tokenizer.fit_on_texts(category_dummy)
#%%to save the tokenizer for the deployment purpose

TOKENIZER_JSON_PATH=os.path.join(os.getcwd(),'tokenizer_new.json')
token_json=tokenizer.to_json()


with open(TOKENIZER_JSON_PATH,'w')as json_file:
    json.dump(token_json,json_file)


#to observe number of words
word_index=tokenizer.word_index
print(word_index)
print(dict(list(word_index.items())[0:5]))


#to vectorize the sequences of words
#vectorize the sequences of text
category_dummy=tokenizer.texts_to_sequences(category_dummy)



temp=([np.shape(i)for i in category_dummy])
np.mean(temp)
category_dummy=pad_sequences(category_dummy,maxlen=100,
                             padding='post',truncating='post')


#OneHotEncoder
#category_dummy=np.reshape(np.array(category_dummy),(1,222500))

#MinMaxScaler


ohe=OneHotEncoder(sparse=False)
text_encoded=ohe.fit_transform(np.expand_dims(text_dummy,axis=-1))
pickle.dump(ohe,open('text_scaler.pkl','wb'))

#train_test_split
X_train,X_test,y_train,y_test=train_test_split(category_dummy,text_encoded,
                                               test_size=0.3,random_state=123)

X_train=np.expand_dims(X_train, axis=-1)
X_test=np.expand_dims(X_test, axis=-1)
print(ohe.inverse_transform(np.expand_dims(y_train[0],axis=0)))

 
#%%MODEL CREATION


model=Sequential()
model.add(Embedding(num_words,64))
model.add(Bidirectional(LSTM(32,return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1,activation='softmax'))


model.summary()
#%%CALLBACKS

log_dir=os.path.join(PATH_LOGS,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

#%%compile&model training

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

model.fit(X_train,y_train,epochs=1,validation_data=(X_test,y_test),callbacks=tensorboard_callback)

#%%model evaluation'

predicted_advanced=np.empty([len(X_test),2])
for index,test in enumerate(X_test):
    predicted_advanced[index,:]=model.predict(np.expand_dims(test, axis=0))
 

#%%model analysis
y_pred=np.argmax(predicted_advanced,axis=1)
y_true=np.argmax(y_test,axis=1)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))


#%%model saving

model.save(MODEL_SAVE_PATH)