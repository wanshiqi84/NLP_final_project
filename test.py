import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed

# Read data from the file
sentences = []
labels = []

with open('english/en_fce_train.tsv', 'r', encoding='utf-8') as f:
    words = []
    curr_labels = []
    for line in f:
        if line.isspace():
            sentences.append(' '.join(words))
            labels.append(' '.join(curr_labels))
            words = []
            curr_labels = []
        else:
            word, label = line.strip().split('\t')
            words.append(word)
            curr_labels.append(label)

# Tokenize the sentences
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# Convert sentences to sequences of indices
sequences = tokenizer.texts_to_sequences(sentences)

# Pad sequences to ensure fixed length
max_length = 200
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Tokenize the labels
label_tokenizer = Tokenizer(oov_token='<OOV>')
label_tokenizer.fit_on_texts(labels)
label_index = label_tokenizer.word_index

# Convert labels to sequences of indices
label_sequences = label_tokenizer.texts_to_sequences(labels)

# Pad label sequences to ensure fixed length
padded_label_sequences = pad_sequences(label_sequences, maxlen=max_length, padding='post', truncating='post')

model = Sequential([
    Embedding(len(word_index) + 1, 128, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    TimeDistributed(Dense(len(label_index) + 1, activation='softmax'))
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train = np.expand_dims(padded_label_sequences, -1)
history = model.fit(padded_sequences, y_train, epochs=5)

# Define test_sentences and test_labels lists
test_sentences = []
test_labels = []

with open('english/en_fce_dev.tsv', 'r', encoding='utf-8') as f:
    words = []
    curr_labels = []
    for line in f:
        if line.strip() == '':
            test_sentences.append(' '.join(words))
            test_labels.append(' '.join(curr_labels))
            words = []
            curr_labels = []
        else:
            word, label = line.strip().split('\t')
            words.append(word)
            curr_labels.append(label)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Generate predictions
predictions = model.predict(padded_test_sequences)
predicted_labels = np.argmax(predictions, axis=-1)

# Filter out the padding tokens
predicted_labels_filtered = np.where(predicted_labels == 0, 0, predicted_labels)

# Convert filtered label sequences to text
predicted_labels_text = label_tokenizer.sequences_to_texts(predicted_labels_filtered)

with open('english/predictions.tsv', 'w', encoding='utf-8') as f:
    for sent, labels in zip(test_sentences, predicted_labels_text):
        words = sent.split()
        pred_labels = labels.split()
        for word, label in zip(words, pred_labels):
            f.write(f"{word}\t{label}\n")
        f.write("\n")
