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
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="auto",
)

# Add description and title
st.write("""
# How Positive or Negative is your Text?
Enter some text and we'll tell you if it has a positive, negative, or neutral sentiment!
""")


# Add image
image = st.image("https://i0.wp.com/thedatascientist.com/wp-content/uploads/2018/10/sentiment-analysis.png", width=400)

# Get user input
text = st.text_input("Enter some text here:")

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


# Show sentiment output
if text:
    sentiment = predict_sentiment(text)
    if sentiment == "Positive":
        st.success(f"The sentiment is {sentiment}!")
    elif sentiment == "Negative":
        st.error(f"The sentiment is {sentiment}.")
    else:
        st.warning(f"The sentiment is {sentiment}.")
