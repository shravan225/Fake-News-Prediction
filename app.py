import streamlit as st
import joblib
import re
import string

# Load models and vectorizer
lr_model = joblib.load(r'C:\Users\samal\Downloads\task_4\logistic_regression_model.pkl')
dt_model = joblib.load(r'C:\Users\samal\Downloads\task_4\decision_tree_model.pkl')
gbc_model = joblib.load(r'C:\Users\samal\Downloads\task_4\gradient_boosting_model.pkl')
rfc_model = joblib.load(r'C:\Users\samal\Downloads\task_4\random_forest_model.pkl')
vectorizer = joblib.load(r'C:\Users\samal\Downloads\task_4\tfidf_vectorizer.pkl')

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

# Output labels
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Streamlit app
def main():
    st.title("Fake News Detection System")
    st.subheader("Enter the news content below to check its authenticity")
    
    # Text input
    news_text = st.text_area("News Text", "", height=200)
    
    if st.button("Analyze"):
        if news_text:
            # Preprocess the text
            processed_text = wordopt(news_text)
            
            # Vectorize the text
            vectorized_text = vectorizer.transform([processed_text])
            
            # Make predictions
            lr_pred = lr_model.predict(vectorized_text)
            dt_pred = dt_model.predict(vectorized_text)
            gbc_pred = gbc_model.predict(vectorized_text)
            rfc_pred = rfc_model.predict(vectorized_text)
            
            # Display results
            st.subheader("Prediction Results")
            st.write(f"Logistic Regression: {output_lable(lr_pred[0])}")
            st.write(f"Decision Tree: {output_lable(dt_pred[0])}")
            st.write(f"Gradient Boosting: {output_lable(gbc_pred[0])}")
            st.write(f"Random Forest: {output_lable(rfc_pred[0])}")
            
            # Show confidence scores (for LR as example)
            if hasattr(lr_model, "predict_proba"):
                lr_proba = lr_model.predict_proba(vectorized_text)
                st.write(f"\nConfidence Scores (Logistic Regression):")
                st.write(f"Fake News: {lr_proba[0][0]*100:.2f}%")
                st.write(f"Not Fake News: {lr_proba[0][1]*100:.2f}%")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == '__main__':
    main()