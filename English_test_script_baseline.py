import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense

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

# Print first sentence and its corresponding label sequence
print(sentences[0])
print(labels[0])

# Initialize the tokenizer without any filters
tokenizer = Tokenizer(oov_token='<OOV>', filters='')
tokenizer.fit_on_texts(sentences)

# Create word-to-index dictionary
word_index = tokenizer.word_index

# Convert sentences to sequences of indices
sequences = tokenizer.texts_to_sequences(sentences)

# Print the first sequence
print(sequences[0])

# Pad sequences to ensure fixed length
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Print the first padded sequence
print(padded_sequences[0])

# Plot the sentence length distribution
plt.hist([len(s.split()) for s in sentences], bins=50)
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.title('Sentence Length Distribution')
plt.show()

# Tokenize the labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

# Create label-to-index dictionary
label_index = label_tokenizer.word_index

# Convert labels to sequences of indices
label_sequences = label_tokenizer.texts_to_sequences(labels)

# Pad label sequences to ensure fixed length
padded_label_sequences = pad_sequences(label_sequences, maxlen=max_length, padding='post', truncating='post')

# Create one-hot encoded label sequences
one_hot_label_sequences = np.zeros((len(padded_label_sequences), max_length, len(label_index) + 1))
for i, seq in enumerate(padded_label_sequences):
    for j, label in enumerate(seq):
        one_hot_label_sequences[i, j, label] = 1

# Split the data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(padded_sequences, one_hot_label_sequences,
                                                                            test_size=0.2)

# Create the FFNN model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=max_length),
    Dense(64, activation='relu'),
    Dense(len(label_index) + 1, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_sentences, train_labels, epochs=5, validation_data=(val_sentences, val_labels))

# Evaluate the model
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
predicted_labels_text = label_tokenizer.sequences_to_texts(predicted_labels)

print("Number of test sentences:", len(test_sentences))
print("Number of predicted labels:", len(predicted_labels_text))
assert len(test_sentences) == len(predicted_labels_text)
# Count the number of differences between predicted labels and actual labels
num_diffs = 0
for pred_labels, actual_labels in zip(predicted_labels, label_sequences):
    for pred_label, actual_label in zip(pred_labels, actual_labels):
        if pred_label != actual_label:
            num_diffs += 1

# Print the number of differences
print("Number of differences between predicted labels and actual labels:", num_diffs)
line_counter = 1
with open('english/predictions.tsv', 'w', encoding='utf-8') as f:
    for sent, labels, original_seq in zip(test_sentences, predicted_labels_text, test_sequences):
        # Use the original words directly
        words = sent.split()

        pred_labels = labels.split()

        # Create a mask based on the original test sequence (without padding)
        mask = [token != 0 for token in original_seq]

        # Filter out padding from words and predicted labels
        filtered_words = [word for word, m in zip(words, mask) if m]
        filtered_labels = [label for label, m in zip(pred_labels, mask) if m]
        while len(filtered_labels) < len(filtered_words):
            filtered_labels.append('c')  # Append 'c' (correct) to match the length

        if len(filtered_words) != len(filtered_labels):
            print(f"Line {line_counter}: Length mismatch")
            print(f"Original sentence: {sent}")
            print(f"Filtered words: {' '.join(filtered_words)}")
            print(f"Filtered labels: {' '.join(filtered_labels)}")

        for word, label in zip(filtered_words, filtered_labels):
            f.write(f"{word}\t{label}\n")
        f.write("\n")

        line_counter += 1

