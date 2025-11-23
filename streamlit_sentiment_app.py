"""
Streamlit App for Sentiment Analysis
Uses three ML models: Logistic Regression, SVM, and Naive Bayes
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path

# ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# NLTK (if needed for preprocessing)
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Text preprocessing function (same as in notebook)
def text_processing(text):
    """Preprocess text for sentiment analysis"""
    # Step 1: Convert to lowercase
    text = str(text).lower()
    # Step 2: Remove URLs
    text = re.sub(r"http\S+", "", text)    
    # Step 3: Remove Mentions and hashtags
    text = re.sub(r"@\w+", "", text)     
    # Step 4: Remove punctuations and special characters
    text = re.sub(r"[^a-zA-Z0-9\s!?']", "", text)   
    text = re.sub(r"\s+", " ", text).strip()        
    return text

@st.cache_resource
def load_or_train_models():
    """
    Load pre-trained models if available, otherwise train new ones
    """
    models = {}
    model_names = {
        'logistic_regression': 'Logistic Regression',
        'svm': 'Support Vector Machine',
        'naive_bayes': 'Naive Bayes'
    }
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Check if all models exist
    all_models_exist = all((models_dir / f'{name}.pkl').exists() 
                          for name in model_names.keys())
    
    if all_models_exist:
        st.info("📦 Loading pre-trained models...")
        try:
            for model_key, model_name in model_names.items():
                with open(models_dir / f'{model_key}.pkl', 'rb') as f:
                    models[model_key] = pickle.load(f)
            st.success("✅ Models loaded successfully!")
            return models
        except Exception as e:
            st.warning(f"⚠️ Error loading models: {e}. Will train new models...")
    
    # Train new models if they don't exist or failed to load
    st.info("🤖 Training models (this may take a few minutes)...")
    
    # Load dataset
    data_path = 'dataset.csv'
    try:
        df = pd.read_csv(data_path, encoding='latin-1', header=None)
        df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
        st.write(f"📊 Loaded {len(df)} tweets from dataset")
    except FileNotFoundError:
        st.error(f"❌ Dataset file '{data_path}' not found. Please ensure the dataset is in the same directory.")
        st.stop()
    
    # Preprocess data
    with st.spinner("🔄 Preprocessing data..."):
        df = df[["text", "target"]]
        df['text'] = df['text'].apply(text_processing)
        
        # Sample data for faster training
        sample_result = train_test_split(
            df, 
            stratify=df["target"], 
            train_size=100000,  # Reduced for faster training in Streamlit
            random_state=42
        )
        df_sampled = sample_result[0]
        X, y = df_sampled["text"], df_sampled["target"]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        st.write(f"✅ Preprocessed {len(X_train)} training samples")
    
    # Train models
    tfidf_vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
    
    # Logistic Regression
    with st.spinner("🤖 Training Logistic Regression model..."):
        pipeline_lr = Pipeline([
            ("tfidf", tfidf_vectorizer),
            ("lr", LogisticRegression(C=0.7, penalty="l2", max_iter=1000, random_state=42))
        ])
        pipeline_lr.fit(X_train, y_train)
        models['logistic_regression'] = pipeline_lr
        st.success("✅ Logistic Regression trained!")
    
    # SVM
    with st.spinner("🤖 Training SVM model..."):
        pipeline_svm = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1, 2))),
            ("svc", LinearSVC(C=0.5, max_iter=1000, random_state=42))
        ])
        pipeline_svm.fit(X_train, y_train)
        models['svm'] = pipeline_svm
        st.success("✅ SVM trained!")
    
    # Naive Bayes
    with st.spinner("🤖 Training Naive Bayes model..."):
        pipeline_nb = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1, 2))),
            ("nb", MultinomialNB(alpha=1.0))
        ])
        pipeline_nb.fit(X_train, y_train)
        models['naive_bayes'] = pipeline_nb
        st.success("✅ Naive Bayes trained!")
    
    # Save models
    with st.spinner("💾 Saving models..."):
        for model_key in models.keys():
            with open(models_dir / f'{model_key}.pkl', 'wb') as f:
                pickle.dump(models[model_key], f)
        st.success("💾 Models saved successfully!")
    
    return models

def predict_sentiment(text, models):
    """Get predictions from all models"""
    # Preprocess input text
    processed_text = text_processing(text)
    
    predictions = {}
    
    for model_key, model in models.items():
        try:
            pred = model.predict([processed_text])[0]
            # Map 0 to Negative, 4 to Positive (or 1 if already mapped)
            if pred == 0:
                sentiment = "Negative"
                sentiment_emoji = "😞"
            else:
                sentiment = "Positive"
                sentiment_emoji = "😊"
            
            # Get prediction probability if available
            try:
                proba = model.predict_proba([processed_text])[0]
                confidence = max(proba) * 100
            except:
                # For SVM (no predict_proba), use decision function
                try:
                    decision = model.decision_function([processed_text])[0]
                    confidence = (abs(decision) / 2) * 100  # Approximate confidence
                    if confidence > 100:
                        confidence = 100
                except:
                    confidence = None
            
            predictions[model_key] = {
                'sentiment': sentiment,
                'emoji': sentiment_emoji,
                'confidence': confidence,
                'raw_prediction': int(pred)
            }
        except Exception as e:
            predictions[model_key] = {
                'sentiment': 'Error',
                'emoji': '❌',
                'confidence': None,
                'error': str(e)
            }
    
    return predictions

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">😊 Sentiment Analysis App</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load or train models
    with st.container():
        models = load_or_train_models()
    
    if not models:
        st.error("Failed to load or train models. Please check the error messages above.")
        st.stop()
    
    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app analyzes sentiment of tweets using three machine learning models:
        
        - **Logistic Regression**
        - **Support Vector Machine (SVM)**
        - **Naive Bayes**
        
        Enter a tweet below and see what each model predicts!
        """)
        
        st.header("📊 Model Information")
        st.write(f"✅ {len(models)} models loaded")
        
        # Example tweets
        st.header("💡 Example Tweets")
        example_tweets = [
            "I love this product! It's amazing! 😊",
            "This is terrible. Worst experience ever.",
            "Had a great day at the park today!",
            "I'm so frustrated with this service."
        ]
        
        for example in example_tweets:
            if st.button(f"📝 {example[:30]}...", key=example):
                st.session_state.example_tweet = example
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Your Tweet")
        
        # Text input
        default_text = st.session_state.get('example_tweet', '')
        user_input = st.text_area(
            "Type or paste a tweet here:",
            value=default_text,
            height=150,
            placeholder="Example: I'm so happy about this amazing product! 🎉"
        )
        
        # Clear session state after using example
        if 'example_tweet' in st.session_state:
            del st.session_state.example_tweet
        
        # Predict button
        predict_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        st.header("📈 Quick Stats")
        st.info("""
        **How it works:**
        1. Enter your tweet
        2. Click "Analyze Sentiment"
        3. See predictions from all 3 models
        """)
    
    # Make predictions
    if predict_button and user_input.strip():
        # Preprocess and predict
        predictions = predict_sentiment(user_input, models)
        
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Display original text
        with st.expander("📄 Original Tweet", expanded=False):
            st.write(f"**Text:** {user_input}")
            st.write(f"**Preprocessed:** {text_processing(user_input)}")
        
        # Create columns for each model
        cols = st.columns(3)
        
        model_display_names = {
            'logistic_regression': 'Logistic Regression',
            'svm': 'SVM',
            'naive_bayes': 'Naive Bayes'
        }
        
        for idx, (model_key, result) in enumerate(predictions.items()):
            with cols[idx]:
                st.markdown(f'<div class="model-card">', unsafe_allow_html=True)
                st.subheader(f"{result['emoji']} {model_display_names[model_key]}")
                
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Sentiment
                    sentiment_class = "positive" if result['sentiment'] == 'Positive' else "negative"
                    st.markdown(
                        f'<p class="{sentiment_class}" style="font-size: 1.5rem;">'
                        f'{result["sentiment"]} {result["emoji"]}</p>',
                        unsafe_allow_html=True
                    )
                    
                    # Confidence
                    if result['confidence'] is not None:
                        st.progress(result['confidence'] / 100)
                        st.caption(f"Confidence: {result['confidence']:.2f}%")
                    else:
                        st.caption("Confidence: N/A")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Consensus prediction
        st.markdown("---")
        st.subheader("🎯 Consensus Prediction")
        
        positive_count = sum(1 for p in predictions.values() 
                            if p.get('sentiment') == 'Positive')
        negative_count = sum(1 for p in predictions.values() 
                            if p.get('sentiment') == 'Negative')
        
        if positive_count > negative_count:
            consensus = "Positive 😊"
            consensus_color = "#28a745"
        elif negative_count > positive_count:
            consensus = "Negative 😞"
            consensus_color = "#dc3545"
        else:
            consensus = "Mixed 🤔"
            consensus_color = "#ffc107"
        
        st.markdown(
            f'<div style="text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 10px;">'
            f'<h2 style="color: {consensus_color};">{consensus}</h2>'
            f'<p>Models agree: {max(positive_count, negative_count)} out of {len(predictions)} models</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Model agreement visualization
        st.markdown("---")
        st.subheader("📊 Model Agreement")
        
        agreement_data = pd.DataFrame({
            'Model': [model_display_names[k] for k in predictions.keys()],
            'Prediction': [p['sentiment'] for p in predictions.values()],
            'Confidence': [p.get('confidence', 0) or 0 for p in predictions.values()]
        })
        
        st.dataframe(agreement_data, use_container_width=True, hide_index=True)
        
    elif predict_button:
        st.warning("⚠️ Please enter a tweet to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with ❤️ using Streamlit | Trained on Twitter Sentiment Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

