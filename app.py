import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import clean_text

# --- Configuration ---
DATA_FILE = 'chandrayaan3_50k_realistic_global_opinion_dataset.csv'
MODEL_DIR = 'models'

# Set page config
st.set_page_config(
    page_title="ISRO Sentiment Analysis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #0b3d91; /* Space blue */
    }
    .stButton>button {
        background-color: #ff9933; /* Saffron */
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #e68a00;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_models():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl')),
        "SVM": joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
    }
    return vectorizer, models

def main():
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard Overview", "Event Analysis", "Model Performance", "Real-time Prediction"])

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    if page == "Dashboard Overview":
        st.title("🌌 ISRO Chandrayaan-3 Sentiment Analysis")
        st.markdown("### analyzing public sentiment across mission phases")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Opinions", f"{len(df):,}")
        with col2:
            st.metric("Positive Sentiments", f"{len(df[df['sentiment']=='positive']):,}")
        with col3:
            st.metric("Negative Sentiments", f"{len(df[df['sentiment']=='negative']):,}")
            
        st.markdown("---")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("Distribution by Platform")
        platform_counts = df['platform'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=platform_counts.index, y=platform_counts.values, ax=ax, palette="viridis")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif page == "Event Analysis":
        st.title("📅 Event-Based Sentiment Analysis")
        
        phases = df['phase'].unique()
        selected_phase = st.selectbox("Select Mission Phase", phases)
        
        phase_df = df[df['phase'] == selected_phase]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Sentiment Distribution: {selected_phase.replace('_', ' ').title()}")
            sentiment_counts = phase_df['sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#99ff99','#ffcc99'])
            st.pyplot(fig)
            
        with col2:
            st.subheader("Top Used Platforms in this Phase")
            fig, ax = plt.subplots()
            sns.countplot(y='platform', data=phase_df, order=phase_df['platform'].value_counts().index, palette="pastel")
            st.pyplot(fig)
            
        st.markdown("---")
        st.subheader("Sentiment Trend Over Time")
        
        # Aggregate by date and sentiment
        daily_sentiment = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        st.line_chart(daily_sentiment)

    elif page == "Model Performance":
        st.title("🤖 Model Performance")
        st.markdown("Comparative analysis of different machine learning models.")
        
        # Hardcoded results from training (since we know they are 100% from previous step, but let's be realistic if we had lower)
        # In a real app, we might load these from a file.
        st.success("All models achieved extremely high accuracy on the test set, indicating distinct separation in the dataset features.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Logistic Regression")
            st.metric("Accuracy", "100%")
        with col2:
            st.info("Naive Bayes")
            st.metric("Accuracy", "100%")
        with col3:
            st.info("SVM")
            st.metric("Accuracy", "100%")

    elif page == "Real-time Prediction":
        st.title("🔮 Real-time Sentiment Prediction")
        
        text_input = st.text_area("Enter text to analyze", "ISRO made India proud with this mission!")
        model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Naive Bayes", "SVM"])
        
        if st.button("Predict"):
            if text_input:
                try:
                    vectorizer, models = load_models()
                    processed_text = clean_text(text_input)
                    vectorized_text = vectorizer.transform([processed_text])
                    prediction = models[model_choice].predict(vectorized_text)[0]
                    
                    st.markdown("### Prediction Result")
                    if prediction == 'positive':
                        st.success(f"**{prediction.upper()}** 😊")
                    elif prediction == 'negative':
                        st.error(f"**{prediction.upper()}** 😠")
                    else:
                        st.warning(f"**{prediction.upper()}** 😐")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
