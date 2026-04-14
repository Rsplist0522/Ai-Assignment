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

# ── Galaxy Theme ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 50%, #0d0221 0%, #060714 40%, #000005 100%) !important;
    color: #e2d9f3 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 60%, rgba(200,180,255,0.6) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 55% 15%, rgba(255,255,255,0.9) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 80%, rgba(180,160,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 40%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 45% 90%, rgba(200,200,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 65% 50%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 85% 10%, rgba(220,200,255,0.8) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 20% 75%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 5% 45%, rgba(200,180,255,0.5) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: rgba(13, 2, 33, 0.85) !important;
    border-right: 1px solid rgba(150, 100, 255, 0.3) !important;
    backdrop-filter: blur(12px);
}

h1 {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.8rem !important;
    background: linear-gradient(135deg, #c77dff, #7b2fff, #e0aaff) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: 2px;
}

h2, h3 {
    font-family: 'Orbitron', monospace !important;
    color: #c77dff !important;
    letter-spacing: 1px;
}

p, label, div { font-family: 'Rajdhani', sans-serif !important; color: #ccc0e8 !important; }

textarea {
    background: rgba(30, 10, 60, 0.8) !important;
    border: 1px solid rgba(150, 80, 255, 0.4) !important;
    border-radius: 10px !important;
    color: #e2d9f3 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    transition: border-color 0.3s ease;
}
textarea:focus {
    border-color: rgba(199, 125, 255, 0.8) !important;
    box-shadow: 0 0 15px rgba(150, 80, 255, 0.3) !important;
}

[data-testid="stButton"] button {
    background: linear-gradient(135deg, #7b2fff, #c77dff) !important;
    color: white !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 1.5px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(123, 47, 255, 0.4) !important;
}
[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(199, 125, 255, 0.6) !important;
}

[data-testid="stSuccess"] {
    background: rgba(30, 80, 50, 0.4) !important;
    border: 1px solid rgba(80, 220, 130, 0.4) !important;
    border-radius: 10px !important;
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 1px;
}
[data-testid="stInfo"] {
    background: rgba(20, 20, 80, 0.5) !important;
    border: 1px solid rgba(100, 120, 255, 0.4) !important;
    border-radius: 10px !important;
}
[data-testid="stWarning"] {
    background: rgba(80, 50, 10, 0.5) !important;
    border: 1px solid rgba(255, 180, 50, 0.4) !important;
    border-radius: 10px !important;
}
[data-testid="stError"] {
    background: rgba(80, 10, 20, 0.5) !important;
    border: 1px solid rgba(255, 80, 100, 0.4) !important;
    border-radius: 10px !important;
}

[data-testid="stMetric"] {
    background: rgba(25, 8, 55, 0.7) !important;
    border: 1px solid rgba(150, 80, 255, 0.35) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    backdrop-filter: blur(8px);
    transition: border-color 0.3s ease;
}
[data-testid="stMetric"]:hover { border-color: rgba(199, 125, 255, 0.6) !important; }
[data-testid="stMetricLabel"] { color: #a08cc8 !important; font-family: 'Rajdhani', sans-serif !important; }
[data-testid="stMetricValue"] { color: #e0aaff !important; font-family: 'Orbitron', monospace !important; font-size: 1rem !important; }

[data-testid="stRadio"] label { color: #ccc0e8 !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #060714; }
::-webkit-scrollbar-thumb { background: rgba(123, 47, 255, 0.5); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

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

# ── Title ──
st.title("🏨 Hotel Review Sentiment Analyzer")
st.write("Enter a hotel review below and click **Analyze** to see the sentiment.")

# ── Mode selector ──
mode = st.sidebar.radio("Analysis Mode", ["🤖 BERT Only", "📊 Compare All Models"])

if mode == "📊 Compare All Models":
    if not all(os.path.exists(p) for p in [NB_MODEL_PATH, SVM_MODEL_PATH, TFIDF_PATH]):
        st.error("NB / SVM models not found. Please train them first.")
        st.stop()
    nb_model, svm_model, tfidf = load_nb_svm()

bert_model, bert_tokenizer = load_bert()
st.success("✦ Model loaded successfully")

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