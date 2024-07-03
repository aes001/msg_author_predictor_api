# This script generates a dummy model for testing purposes.
# The model is not trained. Might be an over complicated script but 
# it's because I was lazy and copied the code from an old project.

import tensorflow as tf
import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np

# Load the dataset
data = pd.read_csv('testData/dummyData.csv')
# data2 = pd.read_csv('Private/HNC_Work_cleaned.csv')

# Concatenate the two datasets
# data = pd.concat([data, data2])
author = data['author']
message = data['message']
header = data.columns

# Count the number of unique authors
author_id = author.unique()
print('Number of unique authors:', len(author_id))


tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(message)
message_sequence = tokenizer.texts_to_sequences(message)

# Pad the tokenized message sequence
max_len = max([len(seq) for seq in message_sequence])
print('Max length of message sequence:', max_len)
message_sequence = tf.keras.utils.pad_sequences(message_sequence, maxlen=max_len)

author_label = tf.keras.utils.to_categorical(author)
print(author_label[0])

# Count how many messages per author
# First create a dictionary of authors and int
author_id = author.unique() 
author_to_messages_count = {auth_id: 0 for auth_id in author_id}

# Count the number of messages per author
for a in author_label:
    author_to_messages_count[np.argmax(a)] += 1

print("Sample size:", len(author_label))
print('Number of messages per author before oversampling:', author_to_messages_count)

# Oversample the dataset
smote = SMOTE()
message_os, author_os = smote.fit_resample(message_sequence, author_label)

# Count the number of messages per author after oversampling
author_to_messages_count = {auth_id: 0 for auth_id in author_id}
for a in author_os:
    author_to_messages_count[np.argmax(a)] += 1

print('Number of messages per author after oversampling:', author_to_messages_count)


# Hyperparameters
x = 256
y = 128
z = 64
dropout_val = 0.5

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(5, x, name='embedding'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(y), name='lstm'),
    tf.keras.layers.Dense(z, activation='relu', name='dense'),
    tf.keras.layers.Dropout(dropout_val, name='dropout'),
    tf.keras.layers.Dense(5, activation='softmax', name='output')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.save("dummyModel_model.keras")


tokenizer_json = tokenizer.to_json()
with open("dummyModel" + '_tokenizer.json', 'w') as json_file:
    json_file.write(tokenizer_json)

