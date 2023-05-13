# NLP Final Project

This repository contains the code and resources for the NLP Final Project. The project focuses on grammatical error detection in essays written in different languages. The goal is to build a system that can identify and label tokens as correct or incorrect based on binary classification.

## Project Structure

The repository is organized into the following directories:

- `czech`: Contains the data and code related to the Czech language.
- `english`: Contains the data and code related to the English language.
- `german`: Contains the data and code related to the German language.
- `italian`: Contains the data and code related to the Italian language.
- `swedish`: Contains the data and code related to the Swedish language.
- `predictions`: Contains the final version of the prediction files.
- `predictions_bert`: Contains the final version of the BERT model prediction files.
- `predictions_test`: Contains the final version of the prediction files for test data.
- `demo`: Contains demo files for each language.
- `ipynb_checkpoints`: Contains checkpoints for Jupyter Notebook files.

Additionally, the repository includes the following files:

- `BERT.py`: Training and saving BERT model.
- `predict_bert.py`: Generate prediction using BERT model.
- `general_LSTM.py`: Training and saving LSTM model.
- `general_baseline.py`: Training and saving baseline model.
- `README.md`: This file, providing an overview of the repository.

## Usage

### Baseline and LSTM Models

To train and predict using the baseline and LSTM models, follow these steps:

1. Run the `general_baseline.py` or `general_LSTM.py` script to train and evaluate the model. The script will automatically generate predictions for the development data and save them in the `predictions`(dev data) directory.
2. Zip the `predictions` and `predictions_test` folders into a single zip file.
3. Upload the zip file to the CodaLab platform to obtain the score for model.

### BERT Model

To train and predict using the BERT model, follow these steps:

1. Train the BERT model specific to the language using the appropriate script (e.g., `train_bert_english.py`).
2. Run the `predict_bert.py` script to generate predictions using the trained BERT model. The predictions will be saved in the `predictions_bert` directory.


