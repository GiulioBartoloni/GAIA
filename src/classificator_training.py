"""classificator_training.py

This script trains an XGBoost classifier for cash flow statement classification.
The model is saved as a pickle file with data necessary for vectorization, encoding and 

Dependencies:
    data/training_dataset.csv

Usage:
    python classificator_training.py
"""

import pandas as pd
import pickle
import os
from pathlib import Path

import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


PROJECT_DIR = Path(__file__).parent.parent
TRAINING_DATASET_CSV = PROJECT_DIR / 'data' / 'xgboost' / 'training_dataset.csv'
MODEL_OUTPUT_PATH = PROJECT_DIR / 'models' / 'cash_flow_classifier.pkl'
CONFIDENCE_THRESHOLD = 0.75
FALLBACK_LABEL = "Other"


def main():
    """Train text classifier and save as pickle file.

    This main function executes the full training process:
        - data loading
        - label encoding
        - TF-IDF vectorization
        - XGBoost training
        - saving pipeline

    raises:
        FileNotFoundError: If training dataset file does not exist.
        KeyError: If required columns ('Description', 'Class') are missing from dataset.
        PermissionError: If unable to write to output directory.
        OSError: If file system operations fail.
    Output:
        Save a serialized dictionary containing the trained model, vectorizer, label encoder, confidence threshold and fallback value.
    """

    if not os.path.exists(TRAINING_DATASET_CSV):
        raise FileNotFoundError(f"Training dataset not found: {TRAINING_DATASET_CSV}.")
    
    # Load training dataset
    dataset = pd.read_csv(TRAINING_DATASET_CSV)

    if 'Description' not in dataset.columns or 'Class' not in dataset.columns:
        raise KeyError(f"Dataset must contain 'Description' and 'Class' columns. Found: {list(dataset.columns)}.")
    
    texts = dataset['Description']
    labels = dataset['Class']

    print(f"Loaded training dataset containing {len(texts)} rows with {len(labels.unique())} classes.")

    # Encode labels to numeric
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Use TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=100,   # Limit vocabulary to 100 elements
        ngram_range=(1, 2), # Considers both unigrams and bigrams
        min_df=1    # unigrams/bigrams must appear at least once
    )
    X = vectorizer.fit_transform(texts)

    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100
    )
    model.fit(X, labels_encoded)

    # Save model data
    pipeline = {
        'model': model,
        'vectorizer': vectorizer,
        'label_encoder': le,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'fallback_label': FALLBACK_LABEL
    }

    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        
        # Save pipeline
        with open(MODEL_OUTPUT_PATH, 'wb') as f:
            pickle.dump(pipeline, f)
        
        print(f"Model successfully saved to {MODEL_OUTPUT_PATH}.")
        
    except PermissionError:
        raise PermissionError(f"No permission to write to {MODEL_OUTPUT_PATH}.")
    except OSError as e:
        raise OSError(f"Failed to save model: {e}.")


if __name__ == "__main__":
    main()