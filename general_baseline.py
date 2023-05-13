import re
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

languages = {
    'cs': 'czech/cs_geccc',
    'de': 'german/de_falko-merlin',
    'en_fce': 'english/en_fce',
    'it': 'italian/it_merlin',
    'sv': 'swedish/sv_swell',
}

for lang_code, data_path in languages.items():
    train_file = f'{data_path}_train.tsv'
    dev_file = f'{data_path}_dev.tsv'
    test_file = f'{data_path}_test_unlabelled.tsv'
    prediction_file = f'predictions/{lang_code}_predictions.tsv'
    prediction_file_test = f'predictions_test/{lang_code}_predictions.tsv'

    # Read data from the train_file
    sentences = []
    labels = []

    with open(train_file, 'r', encoding='utf-8') as f:
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

    # Initialize the tokenizer without any filters
    tokenizer = Tokenizer(oov_token='<OOV>', filters='')
    tokenizer.fit_on_texts(sentences)

    # Create word-to-index dictionary
    word_index = tokenizer.word_index

    # Convert sentences to sequences of indices
    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad sequences to ensure fixed length
    max_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

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

    # Build the model
    vocab_size = len(word_index) + 1
    embedding_dim = 128
    input_seq_length = max_length
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_seq_length),
        Dense(len(label_index) + 1, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Prepare the training data
    y_train = np.expand_dims(padded_label_sequences, -1)

    # Train the model
    history = model.fit(padded_sequences, y_train, epochs=5)

    # Read test data from the file
    test_sentences = []
    test_labels = []

    with open(dev_file, 'r', encoding='utf-8') as f:
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
                if word == "ques '":
                    word = "ques_prime"
                words.append(word)
                curr_labels.append(label)

    # Tokenize and pad the test data
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

    # Generate predictions for the test
    predictions = model.predict(padded_test_sequences)
    predicted_labels = np.argmax(predictions, axis=-1)
    predicted_labels_text = label_tokenizer.sequences_to_texts(predicted_labels)

    # Replace "ques_prime" back to "ques '"
    predicted_labels_text = [re.sub(r'ques_prime', r"ques '", text) for text in predicted_labels_text]

    # Save predictions to the prediction_file
    line_counter = 1
    with open(prediction_file, 'w', encoding='utf-8') as f:
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
                if word == "ques_prime":
                    word = "ques '"
                f.write(f"{word}\t{label}\n")
            f.write("\n")

            line_counter += 1

    test_sentences = []
    test_labels = []
    with open(test_file, 'r', encoding='utf-8') as f:
        words = []
        curr_labels = []
        for line in f:
            if line.strip() == '':
                test_sentences.append(' '.join(words))
                test_labels.append(' '.join(curr_labels))
                words = []
                curr_labels = []
            else:
                word, = line.strip().split('\t')
                if word == "ques '":
                    word = "ques_prime"
                words.append(word)
                curr_labels.append(label)

    # Tokenize the test sentences
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

    # Generate predictions
    predictions = model.predict(padded_test_sequences)
    predicted_labels = np.argmax(predictions, axis=-1)
    predicted_labels_text = label_tokenizer.sequences_to_texts(predicted_labels)

    # Replace "ques_prime" back to "ques '"
    predicted_labels_text = [re.sub(r'ques_prime', r"ques '", text) for text in predicted_labels_text]

    # Save predictions to the prediction_file
    line_counter = 1
    #     print(prediction_file_test)
    with open(prediction_file_test, 'w', encoding='utf-8') as f:
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
                if word == "ques_prime":
                    word = "ques '"
                f.write(f"{word}\t{label}\n")
            f.write("\n")

            line_counter += 1
