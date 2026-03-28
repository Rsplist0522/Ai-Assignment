import os
import torch
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from deep_translator import GoogleTranslator

# ── Page setup ──
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🏨")
st.title("🏨 Hotel Review Sentiment Analyzer")
st.write("Enter a hotel review below and click **Analyze** to see the sentiment.")

# ── Load model (cached so it only loads once) ──
BERT_MODEL_DIR = "saved_models/bert_model"

@st.cache_resource
def load_model():
    model     = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_DIR)
    model.eval()
    return model, tokenizer

# Stop the app if no model found
if not os.path.exists(BERT_MODEL_DIR):
    st.error("No trained model found. Please train the model first.")
    st.stop()

model, tokenizer = load_model()
st.success("Model loaded!")

# ── User input ──
review = st.text_area("Your Review:", placeholder="e.g. The room was clean and staff were friendly.")

# ── Predict on button click ──
if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        # 1. Translate the review to English
        try:
            translated_review = GoogleTranslator(source='auto', target='en').translate(review)
            st.info(f"**Translated text:** {translated_review}")
        except Exception as e:
            st.error("Translation failed. Please try again.")
            st.stop()

        # 2. Tokenize the translated review and predict
        encoded = tokenizer(translated_review, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

        with torch.no_grad():
            output     = model(**encoded)
            prediction = torch.argmax(output.logits, dim=1).item()

        # 3. Map number to sentiment
        sentiment_map = {0: "😞 NEGATIVE", 1: "😐 NEUTRAL", 2: "😊 POSITIVE"}
        result        = sentiment_map[prediction]

        st.subheader("Result:")
        st.success(result)