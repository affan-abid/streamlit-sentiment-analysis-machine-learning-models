"""
Script to save trained models from the notebook
Run this after training models in the sentiment-analysis-simple-models.ipynb notebook
"""

import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import re

# Text preprocessing function (same as in notebook)
def text_processing(text):
    """Preprocess text for sentiment analysis"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)    
    text = re.sub(r"@\w+", "", text)     
    text = re.sub(r"[^a-zA-Z0-9\s!?']", "", text)   
    text = re.sub(r"\s+", " ", text).strip()        
    return text

def main():
    """Load data, train models, and save them"""
    
    print("=" * 60)
    print("Model Training and Saving Script")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    print(f"\n📁 Models will be saved to: {models_dir.absolute()}")
    
    # Load dataset
    print("\n📊 Loading dataset...")
    try:
        df = pd.read_csv('dataset.csv', encoding='latin-1', header=None)
        df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
        print(f"✅ Loaded {len(df)} tweets")
    except FileNotFoundError:
        print("❌ Error: dataset.csv not found!")
        print("Please ensure the dataset is in the same directory as this script.")
        return
    
    # Preprocess data
    print("\n🔄 Preprocessing data...")
    df = df[["text", "target"]]
    df['text'] = df['text'].apply(text_processing)
    
    # Sample data
    print("\n📉 Sampling data...")
    sample_result = train_test_split(
        df, 
        stratify=df["target"], 
        train_size=1000000, 
        random_state=42
    )
    df_sampled = sample_result[0]
    X, y = df_sampled["text"], df_sampled["target"]
    
    # Split into train and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    
    print(f"✅ Training set: {len(X_train)} samples")
    print(f"✅ Validation set: {len(X_val)} samples")
    print(f"✅ Test set: {len(X_test)} samples")
    
    # Train and save Logistic Regression
    print("\n🤖 Training Logistic Regression...")
    pipeline_lr = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1, 2))),
        ("lr", LogisticRegression(C=0.7, penalty="l2", max_iter=1000, random_state=42))
    ])
    pipeline_lr.fit(X_train, y_train)
    
    with open(models_dir / 'logistic_regression.pkl', 'wb') as f:
        pickle.dump(pipeline_lr, f)
    print("✅ Logistic Regression saved!")
    
    # Train and save SVM
    print("\n🤖 Training SVM...")
    pipeline_svm = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1, 2))),
        ("svc", LinearSVC(C=0.5, max_iter=1000, random_state=42))
    ])
    pipeline_svm.fit(X_train, y_train)
    
    with open(models_dir / 'svm.pkl', 'wb') as f:
        pickle.dump(pipeline_svm, f)
    print("✅ SVM saved!")
    
    # Train and save Naive Bayes
    print("\n🤖 Training Naive Bayes...")
    pipeline_nb = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1, 2))),
        ("nb", MultinomialNB(alpha=1.0))
    ])
    pipeline_nb.fit(X_train, y_train)
    
    with open(models_dir / 'naive_bayes.pkl', 'wb') as f:
        pickle.dump(pipeline_nb, f)
    print("✅ Naive Bayes saved!")
    
    print("\n" + "=" * 60)
    print("✅ All models saved successfully!")
    print(f"📁 Location: {models_dir.absolute()}")
    print("=" * 60)
    
    # Print model performance summary
    print("\n📊 Model Performance Summary:")
    print("-" * 60)
    
    from sklearn.metrics import accuracy_score
    
    models = {
        'Logistic Regression': pipeline_lr,
        'SVM': pipeline_svm,
        'Naive Bayes': pipeline_nb
    }
    
    for name, model in models.items():
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        print(f"\n{name}:")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()



