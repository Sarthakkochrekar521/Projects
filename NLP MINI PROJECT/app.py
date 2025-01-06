import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('./depression_classification_bert')
tokenizer = BertTokenizer.from_pretrained('./depression_classification_bert')

# Set the model to evaluation mode
model.eval()

# Function to make predictions
def predict(texts):
    # Tokenize the input text
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

    # Move input tensors to the same device as the model
    input_ids = encodings['input_ids'].to(model.device)
    attention_mask = encodings['attention_mask'].to(model.device)

    with torch.no_grad():
        # Make predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get the predicted class indices
    predicted_class_indices = torch.argmax(logits, dim=1).cpu().numpy()

    return predicted_class_indices

# Mapping of class indices to class labels (Adjust according to your classes)
class_labels = {
    0: "Stress",
    1: "Depression",
    2: "Bipolar Disorder",
    3: "Personality Disorder",
    4: "Anxiety",
    5: "Normal text"
}

# Streamlit app
st.title("Mental Health Classification App")

# Get user input
user_input = st.text_area("Enter a sentence to classify:")

if st.button("Classify"):
    if user_input:
        # Get predictions for the user input
        predictions = predict([user_input])

        # Display the results
        predicted_class = predictions[0]
        st.write(f"Text: '{user_input}' | Predicted Class: {class_labels[predicted_class]}")
    else:
        st.write("Please enter a sentence to classify.")