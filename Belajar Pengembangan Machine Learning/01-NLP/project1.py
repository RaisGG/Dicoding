# -*- coding: utf-8 -*-
"""Project1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wbzwJNwH0D6b2SmVIGG7aqKAQOMogVmo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LSTM, Embedding, GRU
import tensorflow as tf

#Baca file csv(file berasal dari kaggle https://www.kaggle.com/uciml/news-aggregator-dataset)
df = pd.read_csv('/content/sample_data/uci-news-aggregator.csv', usecols=['TITLE', 'CATEGORY'])

#melihat struktur 
df.head()
#df.tail()

#melihat jumlah katagori 
df.CATEGORY.value_counts()

"""Karena label kita berupa data kategorikal, maka kita perlu melakukan proses one-hot-encoding"""

df['ca_labels'] = df['CATEGORY'].map({'b':0, 't':1, 'e':2, 'm':3})

df['ca_labels']

#Agar dapat diproses oleh model, kita perlu mengubah nilai-nilai dari dataframe ke dalam tipe data numpy array menggunakan atribut values.
y = df['ca_labels'].values

#melakukan splitting
x_train, x_test, y_train, y_test = train_test_split(df['TITLE'], y,test_size=0.2)

#Kemudian kita ubah setiap kata pada dataset kita ke dalam bilangan numerik dengan fungsi Tokenizer
vocab_size = 2000
embedded_dim = 8
max_len = 120
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(x_train)
len(word_index)

#Setelah tokenisasi selesai, kita perlu membuat mengonversi setiap sampel menjadi sequence.
padded = pad_sequences(sequences,maxlen=max_len, truncating=trunc_type)
padded.shape

testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences, maxlen=max_len)
testing_padded.shape

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedded_dim,input_length=max_len),
    tf.keras.layers.LSTM(15, return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.92):
      print("\nAkurasi telah mencapai >92%!")
      self.model.stop_training = True
callbacks = myCallback()

model.compile(loss='sparse_categorical_crossentropy',metrics=['acc'], optimizer='adam', )

#Bismillah
hist = model.fit(padded,y_train,batch_size=64, epochs=10, validation_data=(testing_padded,y_test), validation_batch_size=64, callbacks=[callbacks])