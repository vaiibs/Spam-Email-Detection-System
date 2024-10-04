import streamlit as st
import pickle
import re
import pandas as pd

# Load the TF-IDF vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app interface
st.title("Spam Email Detection System")

st.write("Analyze emails to detect whether they are spam or not. You can either type an email or upload a file containing emails for analysis.")
# User input

user_input = st.text_area("Enter the email for analysis")

# Predict button
if st.button("Analyze Email"):
    if user_input:
        input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(input_tfidf)

        if prediction == 0:
            st.success('The email is **Not Spam**.')
        else:
            st.warning('The email is **Spam**.')
    else:
        st.write("Please enter email content to analyze.")


st.subheader("Upload CSV or Excel file for bulk email analysis")
uploaded_file = st.file_uploader("Upload a file containing a column named 'Message'", type=["csv", "xlsx"])
if uploaded_file is not None:

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    if 'Message' not in df.columns:
        st.error("The uploaded file must contain a column named 'Message'.")
    else:
        input_tfidf = tfidf.transform(df['Message'])
        df['prediction'] = model.predict(input_tfidf)
        df['spam_label'] = df['prediction'].apply(lambda x: 'Spam' if x == 1 else 'Not Spam')

        output_file = 'spam_predictions.csv'
        df.to_csv(output_file, index=False)

        # Provide download link for the user
        st.success("Predictions have been saved to 'spam_predictions.csv'.")
        st.download_button(label="Download Predictions", data=open(output_file, 'rb').read(), file_name=output_file,
                           mime='text/csv')