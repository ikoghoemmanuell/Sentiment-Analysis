import streamlit as st
import transformers
import torch

# Load the model and tokenizer
model = transformers.AutoModelForSequenceClassification.from_pretrained("ikoghoemmanuell/finetuned_sentiment_model")
tokenizer = transformers.AutoTokenizer.from_pretrained("ikoghoemmanuell/finetuned_sentiment_tokenizer")

# Define the function for sentiment analysis
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    # Pass the tokenized input through the model
    outputs = model(**inputs)
    # Get the predicted class and return the corresponding sentiment
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return "Positive" if predicted_class == 1 else "Negative"

# Create the Streamlit app
st.title("Sentiment Analysis")

text = st.text_input("Enter a text for sentiment analysis")
if text:
    sentiment = predict_sentiment(text)
    st.write("Sentiment:", sentiment)