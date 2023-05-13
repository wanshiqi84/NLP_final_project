import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# Define a dictionary of custom objects for loading the model
custom_objects = {'TFBertModel': TFBertModel}

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

languages = {
    'cs': 'czech/cs_geccc',
    'de': 'german/de_falko-merlin',
    'en_fce': 'english/en_fce',
    'it': 'italian/it_merlin',
    'sv': 'swedish/sv_swell',
}

for lang_code, data_path in languages.items():
    # Load the saved model
    model = tf.keras.models.load_model(f"{lang_code}_bert_model.h5", custom_objects=custom_objects)

    # Define the label map
    label_map = {0: 'c', 1: 'i'}

    # Load the test dataset
    test_sentences = []

    with open(f"{data_path}_dev.tsv", 'r', encoding='utf-8') as f:
        words = []
        for line in f:
            line = line.strip()
            if line == '':
                test_sentences.append(words)
                words = []
            else:
                word, label = line.split('\t')
                words.append(word)

    # Tokenize the test sentences
    test_sentences_str = [' '.join(words) for words in test_sentences]
    encoded_test_inputs = tokenizer(test_sentences_str, padding=True, truncation=True, max_length=128,
                                    return_tensors='tf')

    # Extract input IDs and attention masks
    test_input_ids = encoded_test_inputs['input_ids']
    test_attention_masks = encoded_test_inputs['attention_mask']

    # Generate predictions
    predictions = model.predict([test_input_ids, test_attention_masks])
    predicted_labels = np.argmax(predictions, axis=-1)

    # Convert predicted labels to text format
    predicted_labels_text = []
    for labels in predicted_labels:
        pred_labels_text = [label_map[label] for label in labels]
        predicted_labels_text.append(' '.join(pred_labels_text))

    # Save the predictions to a file
    with open(f"predictions_bert/{lang_code}_predictions.tsv", 'w', encoding='utf-8') as f:
        for sent, labels in zip(test_sentences, predicted_labels_text):
            for word, label in zip(sent, labels.split()):
                f.write(f"{word}\t{label}\n")
            f.write('\n')
