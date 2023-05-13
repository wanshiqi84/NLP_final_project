import re
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed
from transformers import BertTokenizer, TFBertModel

import tensorflow as tf
print(tf.test.is_gpu_available())
import sys
print(sys.executable)

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
    from sklearn.model_selection import train_test_split

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

    # Split data into training and validation sets
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2,
                                                                                random_state=42)

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Tokenize sentences
    encoded_train_inputs = tokenizer(train_sentences, padding=True, truncation=True, max_length=128,
                                     return_tensors='tf')
    encoded_val_inputs = tokenizer(val_sentences, padding=True, truncation=True, max_length=128, return_tensors='tf')

    # Extract input IDs and attention masks
    train_input_ids = encoded_train_inputs['input_ids']
    train_attention_masks = encoded_train_inputs['attention_mask']
    val_input_ids = encoded_val_inputs['input_ids']
    val_attention_masks = encoded_val_inputs['attention_mask']

    # Convert labels to numerical format
    # Create a label to index mapping
    label_map = {'c': 0, 'i': 1}

    # Convert label strings to numerical values
    num_train_labels = []
    for sentence_labels in train_labels:
        num_sentence_labels = [label_map[label] for label in sentence_labels.split()]
        num_train_labels.append(num_sentence_labels)

    num_val_labels = []
    for sentence_labels in val_labels:
        num_sentence_labels = [label_map[label] for label in sentence_labels.split()]
        num_val_labels.append(num_sentence_labels)

    # Pad the numerical labels to the same length as the padded sentences
    max_len = 128
    padded_train_labels = pad_sequences(num_train_labels, maxlen=max_len, padding='post', value=label_map['c'])
    padded_val_labels = pad_sequences(num_val_labels, maxlen=max_len, padding='post', value=label_map['c'])

    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

    # Load BERT model
    bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')

    # Get BERT output
    bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]

    # Add a Dense layer for classification
    output = tf.keras.layers.Dense(2, activation='softmax', name='output')(bert_output)

    # Define the model
    model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        [train_input_ids, train_attention_masks],
        tf.keras.utils.to_categorical(padded_train_labels),
        epochs=1,
        batch_size=5,
        validation_data=([val_input_ids, val_attention_masks], tf.keras.utils.to_categorical(padded_val_labels))
    )
