import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# --- Configuration ---
DATA_FILE = 'chandrayaan3_50k_realistic_global_opinion_dataset.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text) # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Map sentiment to numerical (though models can handle string labels, numbers are safer for some)
    # Actually, sklearn handles string labels fine, but let's see distribution first.
    # The dataset has 'positive', 'negative', 'neutral'.
    
    X = df['cleaned_text']
    y = df['sentiment']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Save vectorizer
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(dual='auto')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print(report)
        
        # Save model
        filename = os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}_model.pkl")
        joblib.dump(model, filename)
        
        results[name] = {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': cm
        }

    # Save test data for potential dashboard use (optional, but good for consistent eval)
    joblib.dump((X_test, y_test), os.path.join(MODEL_DIR, 'test_data.pkl'))
    
    print("Training complete. Models and vectorizer saved.")
    
if __name__ == "__main__":
    main()
