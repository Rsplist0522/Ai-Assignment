import os
import re
import torch
import joblib
import contractions
import emoji
import streamlit as st
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Page setup ──
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🏨")
st.title("🏨 Hotel Review Sentiment Analyzer")
st.write("Enter a hotel review below and click **Analyze** to see the sentiment.")

# ── Constants ──
MODELS_DIR     = "saved_models"
BERT_MODEL_DIR = os.path.join(MODELS_DIR, "bert_model")
NB_MODEL_PATH  = os.path.join(MODELS_DIR, "nb_model.joblib")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.joblib")
TFIDF_PATH     = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

SENTIMENT_MAP  = {0: "😞 NEGATIVE", 1: "😐 NEUTRAL", 2: "😊 POSITIVE"}
STR_TO_NUM     = {"negative": 0, "neutral": 1, "positive": 2}

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

# ── Preprocessing ──
def preprocess_text(text):
    text   = re.sub(r'\d+', '', text.lower())
    text   = re.sub(r'[^a-z\s]', '', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def preprocess_bert(text):
    text = re.sub(r'<.*?>|http\S+', '', str(text).lower())
    text = contractions.fix(text)
    text = emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ").replace(":", "")
    return " ".join(text.split())

# ── Translation ──
def translate_if_needed(text):
    DetectorFactory.seed = 0
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source="auto", target="en").translate(text), lang
    except Exception:
        pass
    return text, None

# ── Model loaders ──
@st.cache_resource
def load_bert():
    model     = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_DIR)
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_nb_svm():
    return joblib.load(NB_MODEL_PATH), joblib.load(SVM_MODEL_PATH), joblib.load(TFIDF_PATH)

# ── Predictions ──
def predict_bert(text, model, tokenizer):
    enc = tokenizer(preprocess_bert(text), max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        return SENTIMENT_MAP[torch.argmax(model(**enc).logits, dim=1).item()]

def predict_nb_svm(text, nb, svm, tfidf):
    vec = tfidf.transform([preprocess_text(text)])
    return (SENTIMENT_MAP[STR_TO_NUM[nb.predict(vec)[0]]],
            SENTIMENT_MAP[STR_TO_NUM[svm.predict(vec)[0]]])

# ── Guard: BERT must exist ──
if not os.path.exists(BERT_MODEL_DIR):
    st.error("No trained BERT model found. Please train the model first.")
    st.stop()

# ── Mode selector ──
mode = st.sidebar.radio("Analysis Mode", ["🤖 BERT Only", "📊 Compare All Models"])

if mode == "📊 Compare All Models":
    if not all(os.path.exists(p) for p in [NB_MODEL_PATH, SVM_MODEL_PATH, TFIDF_PATH]):
        st.error("NB / SVM models not found. Please train them first.")
        st.stop()
    nb_model, svm_model, tfidf = load_nb_svm()

bert_model, bert_tokenizer = load_bert()
st.success("Model loaded!")

# ── Input ──
review = st.text_area("Your Review:", placeholder="e.g. The room was clean and staff were friendly.")

if st.button("Analyze"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        translated, lang = translate_if_needed(review)
        if lang:
            st.info(f"**Translated:** {translated}")

        if mode == "🤖 BERT Only":
            st.subheader("Result:")
            st.success(predict_bert(translated, bert_model, bert_tokenizer))

        else:
            nb_res, svm_res = predict_nb_svm(translated, nb_model, svm_model, tfidf)
            bert_res        = predict_bert(translated, bert_model, bert_tokenizer)

            st.subheader("Results:")
            col1, col2, col3 = st.columns(3)
            col1.metric("Naïve Bayes", nb_res)
            col2.metric("SVM", svm_res)
            col3.metric("BERT", bert_res)

            labels = [nb_res, svm_res, bert_res]
            if len(set(labels)) == 1:
                st.success("✅ All three models agree!")
            elif len(set(labels)) == 2:
                st.info(f"🤝 Majority sentiment: **{Counter(labels).most_common(1)[0][0]}**")
            else:
                st.warning("⚠️ All three models disagree — the review may be ambiguous.")