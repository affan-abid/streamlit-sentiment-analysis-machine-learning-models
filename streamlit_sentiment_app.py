"""
Streamlit App for Sentiment Analysis
Uses three ML models: Logistic Regression, SVM, Naive Bayes, and a Neural Network
"""

import streamlit as st
import pandas as pd
import pickle
import re
import numpy as np
from pathlib import Path

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

# Preprocessing functions for Neural Network (must match the ones used when saving the model)
# These functions are needed for unpickling the model wrapper
def clean_text(text):
    """Clean text for neural network preprocessing (from LSTM section)"""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags (but keep the text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    # Remove special characters but keep basic punctuation for sentiment
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """Preprocess text with stopwords removal and lemmatization (from LSTM section)"""
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download NLTK data if needed (silent)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    # Keep negation words and intensifiers
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Neural Network Model Wrapper Class (must match the one used when saving the model)
# This class must be defined before loading the pickled model
class NeuralNetworkSentimentModel:
    """
    Wrapper class for the neural network model with tokenizer and preprocessing
    Rebuilds model from architecture parameters and loads weights from .h5 file
    """
    def __init__(self, weights_path, tokenizer, max_sequence_length, max_words, 
                 embedding_dim, clean_text_func, preprocess_text_func):
        self.weights_path = weights_path  # Path to .h5 weights file
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.clean_text = clean_text_func
        self.preprocess_text = preprocess_text_func
        self._model = None  # Lazy loading - rebuild model when needed
    
    def _get_model(self):
        """Rebuild model from architecture and load weights"""
        if self._model is None:
            try:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
            except ImportError:
                raise ImportError("TensorFlow is required for neural network predictions. Install with: pip install tensorflow")
            
            # Rebuild the LSTM model architecture (matching training code)
            self._model = Sequential([
                Embedding(input_dim=self.max_words, output_dim=self.embedding_dim, 
                          input_length=self.max_sequence_length, mask_zero=True),
                Dropout(0.3),
                LSTM(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(32, activation='relu'),
                Dropout(0.4),
                Dense(1, activation='sigmoid')
            ])
            
            # Compile the model
            self._model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Load weights
            import os
            if os.path.isabs(self.weights_path):
                weights_file = self.weights_path
            else:
                # Try relative to current directory first
                weights_file = os.path.abspath(self.weights_path)
                if not os.path.exists(weights_file):
                    # Try relative to script directory
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    weights_file = os.path.join(script_dir, self.weights_path)
            
            # Try alternative paths if original doesn't exist
            if not os.path.exists(weights_file):
                alt_paths = [
                    "best_model_weights.h5",
                    "best_model_portable.keras",
                    "best_model.keras",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_weights.h5"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_portable.keras"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.keras"),
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        weights_file = alt_path
                        break
                else:
                    raise FileNotFoundError(
                        f"Weights file not found: {self.weights_path}\n"
                        f"Tried: {weights_file}\n"
                        f"Also tried: {alt_paths}\n"
                        f"Please ensure 'best_model_weights.h5' is in the same directory as the Streamlit app.\n"
                        f"Run the export cell in the notebook to generate this file."
                    )
            
            # Load weights
            try:
                if weights_file.endswith('.h5'):
                    self._model.load_weights(weights_file)
                else:
                    # If it's a .keras file, try loading as full model
                    from tensorflow.keras.models import load_model
                    self._model = load_model(weights_file, compile=False)
                    self._model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            except Exception as e:
                raise Exception(
                    f"Failed to load model weights: {str(e)}\n\n"
                    f"Please ensure you've run the export cell in the notebook to create 'best_model_weights.h5'"
                )
        return self._model
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError:
            raise ImportError("TensorFlow is required for neural network predictions. Install with: pip install tensorflow")
        
        # Clean and preprocess
        cleaned = self.preprocess_text(self.clean_text(text))
        
        # Tokenize and pad
        seq = self.tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=self.max_sequence_length)
        
        # Get model and predict
        model = self._get_model()
        pred = model.predict(pad, verbose=0)[0][0]
        
        # Return sentiment and confidence
        sentiment = 'Positive' if pred > 0.5 else 'Negative'
        confidence = pred if pred > 0.5 else 1 - pred
        
        return sentiment, float(pred), float(confidence)
    
    def predict(self, texts):
        """Predict sentiment for multiple texts (compatible with sklearn interface)"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for text in texts:
            sentiment, prob, conf = self.predict_sentiment(text)
            # Return class label: 1 for Positive, 0 for Negative
            class_label = 1 if sentiment == 'Positive' else 0
            results.append(class_label)
        
        return np.array(results)

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
def load_models():
    """
    Load pre-trained sklearn models from the current directory
    """
    models = {}
    model_names = {
        'logistic_regression': 'Logistic Regression',
        'svm': 'Support Vector Machine',
        'naive_bayes': 'Naive Bayes'
    }
    
    models_dir = Path('./')
    missing_models = []
    
    # Check which models exist
    for model_key in model_names.keys():
        model_path = models_dir / f'{model_key}.pkl'
        if not model_path.exists():
            missing_models.append(model_key)
    
    if missing_models:
        st.error(f"❌ Missing model files: {', '.join(missing_models)}.pkl")
        st.error("Please ensure all model files (.pkl) are in the same directory as this app.")
        st.stop()
        return None
    
    # Load all models
    with st.spinner("📦 Loading pre-trained models..."):
        try:
            for model_key, model_name in model_names.items():
                model_path = models_dir / f'{model_key}.pkl'
                with open(model_path, 'rb') as f:
                    models[model_key] = pickle.load(f)
            st.success(f"✅ Successfully loaded {len(models)} sklearn models!")
            return models
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.error("Please check that the model files are valid pickle files.")
            st.stop()
            return None

@st.cache_resource
def load_neural_network():
    """
    Load the neural network model (best_model_neural_network.pkl)
    """
    model_path = Path('./best_model_neural_network.pkl')
    
    if not model_path.exists():
        st.warning("⚠️ Neural Network model (best_model_neural_network.pkl) not found. Skipping neural network predictions.")
        return None
    
    try:
        with st.spinner("🧠 Loading Neural Network model..."):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Verify the model file path exists
            import os
            model_file_path = os.path.abspath(model.weights_path)
            if not os.path.exists(model_file_path):
                st.error(f"❌ Error: The model file '{model.model_path}' was not found.")
                st.error(f"Expected location: {model_file_path}")
                st.error("Please ensure 'best_model.keras' is in the same directory as this app.")
                return None
            
            st.success("✅ Neural Network model wrapper loaded successfully!")
            st.info(f"📁 Model file: {model.model_path}")
            return model
    except AttributeError as e:
        if "NeuralNetworkSentimentModel" in str(e):
            st.error("❌ Error: The NeuralNetworkSentimentModel class definition is missing or doesn't match the saved model.")
            st.error("Please ensure the class is defined in this file before loading the model.")
        else:
            st.error(f"❌ Error loading Neural Network model: {str(e)}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error loading Neural Network model: {str(e)}")
        return None

def predict_with_neural_network(text, nn_model):
    """Get prediction from neural network model (PKL wrapper)"""
    if nn_model is None:
        return None
    
    try:
        # Use the wrapper's predict_sentiment method
        # The wrapper handles all preprocessing internally
        sentiment, prob, conf = nn_model.predict_sentiment(text)
        
        # Map sentiment to format
        sentiment_emoji = "😊" if sentiment == "Positive" else "😞"
        pred_class = 1 if sentiment == "Positive" else 0
        
        return {
            'sentiment': sentiment,
            'emoji': sentiment_emoji,
            'confidence': conf * 100,  # Convert to percentage
            'raw_prediction': pred_class,
            'raw_probability': float(prob)
        }
    except Exception as e:
        return {
            'sentiment': 'Error',
            'emoji': '❌',
            'confidence': None,
            'error': str(e)
        }

def predict_sentiment(text, models, nn_model=None):
    """Get predictions from all models including neural network"""
    # Preprocess input text for sklearn models
    processed_text = text_processing(text)
    
    predictions = {}
    
    # Get predictions from sklearn models
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
    
    # Get prediction from neural network (uses its own preprocessing)
    if nn_model is not None:
        nn_prediction = predict_with_neural_network(text, nn_model)
        if nn_prediction:
            predictions['neural_network'] = nn_prediction
    
    return predictions

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">😊 Sentiment Analysis App</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    with st.container():
        models = load_models()
        nn_model = load_neural_network()
    
    if not models:
        st.error("Failed to load models. Please check the error messages above.")
        st.stop()
    
    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app analyzes sentiment of tweets using multiple machine learning models:
        
        - **Logistic Regression**
        - **Support Vector Machine (SVM)**
        - **Naive Bayes**
        - **Neural Network** 🧠
        
        Enter a tweet below and see what each model predicts!
        """)
        
        st.header("📊 Model Information")
        st.write(f"✅ {len(models)} sklearn models loaded")
        if nn_model is not None:
            st.write("✅ Neural Network model loaded")
        else:
            st.write("⚠️ Neural Network model not available")
        
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
        3. See predictions from all models
        
        **Note:** 
        - Sklearn models loaded from .pkl files
        - Neural Network loaded from best_model_neural_network.pkl
        """)
    
    # Make predictions
    if predict_button and user_input.strip():
        # Preprocess and predict
        predictions = predict_sentiment(user_input, models, nn_model)
        
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Display original text
        with st.expander("📄 Original Tweet", expanded=False):
            st.write(f"**Text:** {user_input}")
            st.write(f"**Preprocessed:** {text_processing(user_input)}")
        
        # Separate sklearn models and neural network
        sklearn_predictions = {k: v for k, v in predictions.items() if k != 'neural_network'}
        nn_prediction = predictions.get('neural_network')
        
        # Create columns for sklearn models
        st.subheader("📊 Traditional ML Models")
        cols = st.columns(3)
        
        model_display_names = {
            'logistic_regression': 'Logistic Regression',
            'svm': 'SVM',
            'naive_bayes': 'Naive Bayes'
        }
        
        for idx, (model_key, result) in enumerate(sklearn_predictions.items()):
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
        
        # Neural Network Section
        if nn_prediction:
            st.markdown("---")
            st.subheader("🧠 Predictions from Neural Network")
            nn_col = st.columns(1)
            
            with nn_col[0]:
                st.markdown(f'<div class="model-card" style="background-color: #e8f4f8;">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #1f77b4;">🧠 Neural Network Model</h3>', unsafe_allow_html=True)
                
                if 'error' in nn_prediction:
                    st.error(f"Error: {nn_prediction['error']}")
                else:
                    # Sentiment
                    sentiment_class = "positive" if nn_prediction['sentiment'] == 'Positive' else "negative"
                    st.markdown(
                        f'<p class="{sentiment_class}" style="font-size: 2rem; text-align: center;">'
                        f'{nn_prediction["sentiment"]} {nn_prediction["emoji"]}</p>',
                        unsafe_allow_html=True
                    )
                    
                    # Confidence
                    if nn_prediction['confidence'] is not None:
                        st.progress(nn_prediction['confidence'] / 100)
                        st.caption(f"Confidence: {nn_prediction['confidence']:.2f}%")
                        if nn_prediction.get('raw_probability') is not None:
                            st.caption(f"Probability: {nn_prediction['raw_probability']:.4f}")
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
        
        # Prepare data for all models
        model_list = []
        prediction_list = []
        confidence_list = []
        
        for k in sklearn_predictions.keys():
            model_list.append(model_display_names[k])
            prediction_list.append(sklearn_predictions[k]['sentiment'])
            confidence_list.append(sklearn_predictions[k].get('confidence', 0) or 0)
        
        if nn_prediction and 'error' not in nn_prediction:
            model_list.append('Neural Network')
            prediction_list.append(nn_prediction['sentiment'])
            confidence_list.append(nn_prediction.get('confidence', 0) or 0)
        
        agreement_data = pd.DataFrame({
            'Model': model_list,
            'Prediction': prediction_list,
            'Confidence': confidence_list
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

