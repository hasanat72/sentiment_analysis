
# app.py
import streamlit as st
import joblib
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Define the SentimentAnalysisClassifier class (copy-pasted from the notebook)
class SentimentAnalysisClassifier:
    def __init__(self):
        """Initialize the sentiment analysis classifier"""
        # These will be loaded later
        self.vectorizer = None
        self.label_encoder = None
        self.best_model = None
        self.stemmer = PorterStemmer()

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
             nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
             nltk.download('punkt_tab')


    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [self.stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]

        return ' '.join(tokens)

    # Modified to not train/load data automatically
    def load_components(self, model_path, vectorizer_path, label_encoder_path):
        """Load trained model, vectorizer, and label encoder"""
        try:
            self.best_model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoder = joblib.load(label_encoder_path)
            st.success("Model components loaded successfully.")
        except FileNotFoundError:
            st.error("Error: Model components not found. Please ensure 'random_forest_sentiment_model.pkl', 'tfidf_vectorizer.pkl', and 'label_encoder.pkl' are in the same directory.")
        except Exception as e:
            st.error(f"An error occurred while loading model components: {e}")


    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        if self.best_model is None or self.vectorizer is None or self.label_encoder is None:
            st.warning("Model components not loaded.")
            return None

        # Preprocess the text
        processed_text = self.preprocess_text(text)

        if not processed_text:
            return {'sentiment': 'neutral', 'confidence': 1.0, 'probabilities': {label: 1.0/len(self.label_encoder.classes_) for label in self.label_encoder.classes_}}


        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])

        # Predict
        prediction = self.best_model.predict(text_vec)[0]

        # Handle models that don't have predict_proba (like SVC with kernel='linear' by default)
        if hasattr(self.best_model, 'predict_proba'):
             probability = self.best_model.predict_proba(text_vec)[0]
             confidence = max(probability)
             probabilities = dict(zip(self.label_encoder.classes_, probability))
        else:
             # For models without predict_proba, we can't provide confidence or probabilities
             probability = None
             confidence = "N/A" # Not Applicable
             probabilities = {label: "N/A" for label in self.label_encoder.classes_}


        # Decode label
        sentiment = self.label_encoder.inverse_transform([prediction])[0]


        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': probabilities
        }

# --- Streamlit App ---
st.title("Sentiment Analysis App")

# Initialize classifier and load components
# Use st.cache_resource to avoid reloading components on every rerun
@st.cache_resource
def load_all_components():
    classifier_instance = SentimentAnalysisClassifier()
    model_path = 'random_forest_sentiment_model.pkl' # Assuming Random Forest was the best or desired model
    vectorizer_path = 'tfidf_vectorizer.pkl'
    label_encoder_path = 'label_encoder.pkl'
    classifier_instance.load_components(model_path, vectorizer_path, label_encoder_path)
    return classifier_instance

classifier = load_all_components()


# Text input from user
user_input = st.text_area("Enter text for sentiment analysis:", "")

if st.button("Predict Sentiment"):
    if user_input:
        if classifier.best_model and classifier.vectorizer and classifier.label_encoder:
            prediction = classifier.predict_sentiment(user_input)
            if prediction:
                st.write(f"**Predicted Sentiment:** {prediction['sentiment']}")
                st.write(f"**Confidence:** {prediction['confidence']:.4f}" if isinstance(prediction['confidence'], float) else f"**Confidence:** {prediction['confidence']}")

                if isinstance(prediction['probabilities'], dict) and "N/A" not in prediction['probabilities'].values():
                    st.write("**Probabilities:**")
                    for sentiment, prob in prediction['probabilities'].items():
                        st.write(f"- {sentiment.capitalize()}: {prob:.4f}")
            else:
                st.write("Could not perform prediction.")
        else:
            st.warning("Model components are not loaded. Please ensure the files are in the correct directory.")
    else:
        st.warning("Please enter some text to analyze.")
