#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
import torch
from transformers import BertTokenizer, BertModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# random.seed(42)
# tf.random.set_seed(42)
# np.random.seed(42)


# In[2]:


MODEL_NAME = 'monologg/biobert_v1.1_pubmed'
MAX_SEQ_LENGHT = 50


# In[3]:


df = pd.read_excel('Data/drugData.xlsx')
df.head()


# In[4]:


labels = df['MOA'].apply(lambda x: 0 if x == 4 else 1)
NUM_LABEL = len(labels.unique())
labels = to_categorical(labels)


# In[5]:


print('There are {} records in the dataset'.format(df.shape[0]))


# In[6]:


df['text'] = df['text'].apply(lambda x: x.replace('\n\n', ''))


# In[7]:


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert = BertModel.from_pretrained(MODEL_NAME)


# In[8]:


LEARNING_RATE = 0.01


model = tf.keras.Sequential([
        tf.keras.layers.Input((bert.config.hidden_size,), name='input_layer'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(bert.config.hidden_size, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(bert.config.hidden_size/2, activation='relu', name='dense_2'),
        tf.keras.layers.Dense(bert.config.hidden_size/4, activation='relu', name='dense_3'),
        tf.keras.layers.Dense(bert.config.hidden_size/8, activation='relu', name='dense_4'),
        tf.keras.layers.Dense(NUM_LABEL, name='dense_5'),
        tf.keras.layers.Activation('softmax', name='softmax')
])

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()


# In[10]:


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_windows = []

    for text in tqdm(example):
        windows = []
        tokens_a = tokenizer.tokenize(text)
        for i in range(0, len(tokens_a), max_seq_length):
            window = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a[i:i + max_seq_length] + ["[SEP]"])+                                                             [0] * (max_seq_length - len(tokens_a))
            windows.extend(window)
        all_windows.append(windows)
        
    return np.array(all_windows)


# In[11]:


tokenized = convert_lines(df['text'], MAX_SEQ_LENGHT, tokenizer)


# In[12]:


cls_features = []
for tokens in tqdm(tokenized):
    CLS = torch.zeros(1,768)
    for i in range(0, len(tokens), MAX_SEQ_LENGHT):
        batch = tokens[i:i + MAX_SEQ_LENGHT]
        if len(tokens) < MAX_SEQ_LENGHT:
            batch = np.pad(batch, (0,MAX_SEQ_LENGHT-len(batch)), 'constant')
        batch = torch.tensor(batch).unsqueeze(0)
            
        CLS += bert(batch)[0][:,0,:]
        
    cls_features.append(CLS)


# In[13]:


cls_features = [c.tolist() for c in cls_features]
cls_features = np.array(cls_features)
cls_features.shape


# In[14]:


cls_features = cls_features.squeeze(1)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(cls_features, labels, test_size=0.25) 


# In[16]:


model.fit(X_train, y_train, batch_size=15, epochs=50)


# In[17]:


res = model.predict(X_test)
y_pred = np.argmax(res, axis=1)


# In[ ]:


print('The accuracy is {:0.02f}%'.format(accuracy_score(np.argmax(y_test, axis=1), y_pred)*100))


# In[ ]:


sns.heatmap(confusion_matrix(np.argmax(y_test, axis=1), y_pred), annot=True)


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score
print('The precision of the model: {:0.02f}%'.format(precision_score(np.argmax(y_test, axis=1), y_pred)))
print('The recall of the model: {:0.02f}%'.format(recall_score(np.argmax(y_test, axis=1), y_pred)))
print('The f1 score of the model: {:0.02f}%'.format(f1_score(np.argmax(y_test, axis=1), y_pred)))

