import streamlit as st
import transformers
import torch

# Load the model and tokenizer
model = transformers.AutoModelForSequenceClassification.from_pretrained("ikoghoemmanuell/finetuned_sentiment_model")
tokenizer = transformers.AutoTokenizer.from_pretrained("ikoghoemmanuell/finetuned_sentiment_tokenizer")

# Define the function for sentiment analysis
@st.cache_resource
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    # Pass the tokenized input through the model
    outputs = model(**inputs)
    # Get the predicted class and return the corresponding sentiment
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    if predicted_class == 0:
        return "Negative"
    elif predicted_class == 1:
        return "Neutral"
    else:
        return "Positive"

# Setting the page configurations
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon=":speech_balloon:",
    layout="wide",
    initial_sidebar_state="auto",
)

# Allow custom CSS
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define the CSS style for the app
st.markdown(
"""
<style>
body {
    background-color: #f5f5f5;
}
h1 {
    color: #4e79a7;
}
</style>

""",
unsafe_allow_html=True
)

# Add logo to the app
# from PIL import Image
# image = Image.open('logo.jpg')
# st.image(image, caption='Sentiment Analysis App')

# Create the Streamlit app
st.title("Sentiment Analysis")

st.write("This app uses a pre-trained machine learning model to predict the sentiment of a given text. Simply enter a piece of text and the app will classify it as positive, negative or neutral.")

text = st.text_input("")
if text:
    sentiment = predict_sentiment(text)
    st.write("Sentiment:", sentiment)

# Add error handling
if not text:
    st.write("Please enter a text above to analyze.")
elif sentiment is None:
    st.write("Sorry, the model failed to make a prediction for this text. Please try again with a different text.") 

# # Add footer to the app
# st.markdown(
# """
# ---
# Sentiment Analysis App developed by Emmanuel Ikogho Okeoghene
# """,
# unsafe_allow_html=True
# )
