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

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Celestial · Hotel Review Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&display=swap');

/* ── Hide Streamlit chrome ── */
#MainMenu, header[data-testid="stHeader"], footer,
[data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}

/* ── FORCE sidebar always open — hide the collapse button ── */
[data-testid="stSidebarCollapseButton"],
button[kind="header"] {
    display: none !important;
}

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"] {
    background: #080b12 !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2738 !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 36px 24px 32px !important;
}

/* ── Main block container ── */
[data-testid="stAppViewBlockContainer"] {
    max-width: 860px !important;
    padding: 52px 60px !important;
}

/* ── Typography ── */
h1 {
    font-family: 'Sora', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    letter-spacing: -1px !important;
    line-height: 1.1 !important;
    margin-bottom: 10px !important;
}

h2, h3 {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.2em !important;
    color: #4c6ef5 !important;
    margin-top: 40px !important;
    margin-bottom: 14px !important;
}

p, div {
    font-family: 'Sora', sans-serif !important;
}

[data-testid="stMarkdownContainer"] p {
    font-size: 1rem !important;
    color: #7a8ba8 !important;
    line-height: 1.7 !important;
}

/* ── Textarea ── */
textarea {
    background: #111827 !important;
    border: 1.5px solid #1e2d45 !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 300 !important;
    line-height: 1.75 !important;
    padding: 18px 20px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
textarea:focus {
    border-color: #4c6ef5 !important;
    box-shadow: 0 0 0 4px rgba(76, 110, 245, 0.12) !important;
    outline: none !important;
}
textarea::placeholder { color: #2e3f58 !important; }

[data-testid="stTextArea"] label {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #4c6ef5 !important;
    margin-bottom: 10px !important;
}

/* ── Button ── */
[data-testid="stButton"] button {
    background: #4c6ef5 !important;
    color: #ffffff !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
}
[data-testid="stButton"] button:hover {
    background: #3b5bdb !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(76, 110, 245, 0.4) !important;
}
[data-testid="stButton"] button:active {
    transform: scale(0.98) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #111827 !important;
    border: 1.5px solid #1e2d45 !important;
    border-radius: 16px !important;
    padding: 26px 22px !important;
    transition: border-color 0.2s, transform 0.2s !important;
}
[data-testid="stMetric"]:hover {
    border-color: #4c6ef5 !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #4c6ef5 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    color: #ffffff !important;
    margin-top: 8px !important;
}

/* ── Alerts ── */
[data-testid="stSuccess"] {
    background: #071a10 !important;
    border: 1.5px solid #14532d !important;
    border-radius: 14px !important;
    padding: 20px 22px !important;
}
[data-testid="stSuccess"] p,
[data-testid="stSuccess"] div {
    color: #4ade80 !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
}
[data-testid="stInfo"] {
    background: #070f1f !important;
    border: 1.5px solid #1e3a6e !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}
[data-testid="stInfo"] p,
[data-testid="stInfo"] div { color: #93c5fd !important; font-size: 0.92rem !important; }
[data-testid="stWarning"] {
    background: #130f00 !important;
    border: 1.5px solid #422006 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}
[data-testid="stWarning"] p,
[data-testid="stWarning"] div { color: #fbbf24 !important; }
[data-testid="stError"] {
    background: #130000 !important;
    border: 1.5px solid #450a0a !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}
[data-testid="stError"] p,
[data-testid="stError"] div { color: #f87171 !important; }

/* ── Radio ── */
[data-testid="stRadio"] > label {
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #4c6ef5 !important;
    margin-bottom: 14px !important;
}
[data-testid="stRadio"] label span {
    font-size: 0.9rem !important;
    color: #c0cfe0 !important;
    font-weight: 400 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #161f30 !important;
    margin: 36px 0 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #4c6ef5 !important; }

/* ── Columns gap ── */
[data-testid="stHorizontalBlock"] { gap: 16px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 4px; }

/* ════════════ CUSTOM COMPONENTS ════════════ */

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 40px;
}
.sidebar-icon {
    width: 40px;
    height: 40px;
    background: #4c6ef5;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.sidebar-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.3px;
}
.sidebar-subtitle {
    font-family: 'Sora', sans-serif;
    font-size: 0.7rem;
    color: #4c6ef5;
    font-weight: 500;
}
.info-block {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.info-label {
    font-family: 'Sora', sans-serif;
    font-size: 0.55rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #4c6ef5;
    margin-bottom: 10px;
}
.info-item {
    font-family: 'Sora', sans-serif;
    font-size: 0.8rem;
    color: #8899b4;
    line-height: 2;
    display: flex;
    align-items: center;
    gap: 8px;
}
.info-dot {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #2d3f5e;
    flex-shrink: 0;
}
.hero-sub {
    font-family: 'Sora', sans-serif;
    font-size: 1.05rem;
    color: #5a7090;
    font-weight: 300;
    margin-bottom: 28px;
    line-height: 1.6;
}
.ready-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #071a10;
    border: 1.5px solid #14532d;
    color: #4ade80;
    font-family: 'Sora', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 7px 16px;
    border-radius: 100px;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ade80;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74,222,128,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 4px rgba(74,222,128,0); }
}
.stat-strip {
    display: flex;
    gap: 2px;
    margin-bottom: 36px;
}
.stat-item {
    flex: 1;
    background: #111827;
    border: 1px solid #1e2d45;
    padding: 16px 18px;
    border-radius: 4px;
    text-align: center;
}
.stat-item:first-child { border-radius: 12px 4px 4px 12px; }
.stat-item:last-child  { border-radius: 4px 12px 12px 4px; }
.stat-val {
    font-family: 'Sora', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff;
    display: block;
}
.stat-lbl {
    font-family: 'Sora', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #3d5070;
    display: block;
    margin-top: 3px;
}
.verdict {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 22px;
    border-radius: 14px;
    font-family: 'Sora', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    margin-top: 10px;
}
.verdict-icon { font-size: 1.3rem; flex-shrink: 0; }
.verdict.agree    { background: #071a10; border: 1.5px solid #14532d; color: #4ade80; }
.verdict.majority { background: #130f00; border: 1.5px solid #422006; color: #fbbf24; }
.verdict.split    { background: #130000; border: 1.5px solid #450a0a; color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR     = "saved_models"
BERT_MODEL_DIR = os.path.join(MODELS_DIR, "bert_model")
NB_MODEL_PATH  = os.path.join(MODELS_DIR, "nb_model.joblib")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.joblib")
TFIDF_PATH     = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

LABEL_MAP  = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOJI_MAP  = {"Negative": "😞", "Neutral": "😐", "Positive": "😊"}
STR_TO_NUM = {"negative": 0, "neutral": 1, "positive": 2}

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    text   = re.sub(r"\d+", "", text.lower())
    text   = re.sub(r"[^a-z\s]", "", text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def preprocess_bert(text: str) -> str:
    text = re.sub(r"<.*?>|http\S+", "", str(text).lower())
    text = contractions.fix(text)
    text = emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ").replace(":", "")
    return " ".join(text.split())

# ─── Translation ───────────────────────────────────────────────────────────────
def translate_if_needed(text: str):
    DetectorFactory.seed = 0
    try:
        lang = detect(text)
        if lang != "en":
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            return translated, lang
    except Exception:
        pass
    return text, None

# ─── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bert():
    model     = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_DIR)
    model.eval()
    return model, tokenizer

@st.cache_resource(show_spinner=False)
def load_nb_svm():
    nb    = joblib.load(NB_MODEL_PATH)
    svm   = joblib.load(SVM_MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return nb, svm, tfidf

# ─── Prediction helpers ────────────────────────────────────────────────────────
def predict_bert(text: str, model, tokenizer) -> str:
    enc = tokenizer(
        preprocess_bert(text),
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        idx = torch.argmax(model(**enc).logits, dim=1).item()
    return LABEL_MAP[idx]

def predict_nb_svm(text: str, nb, svm, tfidf):
    vec     = tfidf.transform([preprocess_text(text)])
    nb_raw  = nb.predict(vec)[0]
    svm_raw = svm.predict(vec)[0]
    return LABEL_MAP[STR_TO_NUM[nb_raw]], LABEL_MAP[STR_TO_NUM[svm_raw]]

# ─── Guard ─────────────────────────────────────────────────────────────────────
if not os.path.exists(BERT_MODEL_DIR):
    st.error("BERT model not found. Please train the model first (`saved_models/bert_model/`).")
    st.stop()

bert_model, bert_tokenizer = load_bert()

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div class='sidebar-logo'>"
        "  <div class='sidebar-icon'>🔍</div>"
        "  <div>"
        "    <div class='sidebar-title'>Celestial</div>"
        "    <div class='sidebar-subtitle'>Hotel Review AI</div>"
        "  </div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='info-label'>Analysis Mode</div>", unsafe_allow_html=True)
    mode = st.radio(
        "Analysis Mode",
        ["BERT Only", "Compare All Models"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "<div class='info-block'>"
        "  <div class='info-label'>Models</div>"
        "  <div class='info-item'><span class='info-dot'></span>DistilBERT — Transformer</div>"
        "  <div class='info-item'><span class='info-dot'></span>Naïve Bayes — TF-IDF</div>"
        "  <div class='info-item'><span class='info-dot'></span>SVM — RBF Kernel</div>"
        "</div>"
        "<div class='info-block'>"
        "  <div class='info-label'>Dataset</div>"
        "  <div class='info-item'><span class='info-dot'></span>TripAdvisor Hotel Reviews</div>"
        "</div>"
        "<div class='info-block'>"
        "  <div class='info-label'>Languages</div>"
        "  <div class='info-item'><span class='info-dot'></span>Auto-detect &amp; Translate</div>"
        "  <div class='info-item'><span class='info-dot'></span>Powered by Google Translate</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Load NB/SVM if needed ─────────────────────────────────────────────────────
nb_model = svm_model = tfidf = None
if mode == "Compare All Models":
    missing = [p for p in [NB_MODEL_PATH, SVM_MODEL_PATH, TFIDF_PATH] if not os.path.exists(p)]
    if missing:
        st.error(f"Missing model files: {', '.join(missing)}")
        st.stop()
    nb_model, svm_model, tfidf = load_nb_svm()

# ══════════════════════════════════════════════════════
#  MAIN — HEADER
# ══════════════════════════════════════════════════════
st.markdown("# Celestial")
st.markdown(
    "<p class='hero-sub'>Hotel review sentiment analysis · BERT · Naïve Bayes · SVM</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='margin-bottom:36px;'>"
    "<span class='ready-badge'><span class='pulse-dot'></span>All Systems Ready</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='stat-strip'>"
    "  <div class='stat-item'><span class='stat-val'>3</span><span class='stat-lbl'>Models Active</span></div>"
    "  <div class='stat-item'><span class='stat-val'>100+</span><span class='stat-lbl'>Languages</span></div>"
    "  <div class='stat-item'><span class='stat-val'>3</span><span class='stat-lbl'>Sentiment Classes</span></div>"
    "  <div class='stat-item'><span class='stat-val'>128</span><span class='stat-lbl'>Token Window</span></div>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ══════════════════════════════════════════════════════
#  MAIN — INPUT
# ══════════════════════════════════════════════════════
review = st.text_area(
    "Review Text",
    placeholder="Paste any hotel review here — multilingual support included...",
    height=160,
)

col_btn, _, _ = st.columns([1, 1, 2])
with col_btn:
    run = st.button("Analyze →")

# ══════════════════════════════════════════════════════
#  MAIN — ANALYSIS
# ══════════════════════════════════════════════════════
if run:
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
        st.stop()

    with st.spinner("Analyzing…"):
        translated, detected_lang = translate_if_needed(review)

    if detected_lang:
        st.markdown("---")
        st.markdown("### Translation")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Detected Language:** `{detected_lang.upper()}`")
        with c2:
            st.info(f"**Translated Text:** {translated}")

    st.markdown("---")

    if mode == "BERT Only":
        label = predict_bert(translated, bert_model, bert_tokenizer)
        icon  = EMOJI_MAP[label]
        st.markdown("### Result")
        st.success(f"{icon}  **{label}** — sentiment detected by BERT.")

    else:
        nb_label, svm_label = predict_nb_svm(translated, nb_model, svm_model, tfidf)
        bert_label          = predict_bert(translated, bert_model, bert_tokenizer)

        st.markdown("### Model Predictions")
        c1, c2, c3 = st.columns(3)
        c1.metric("Naïve Bayes", f"{EMOJI_MAP[nb_label]}  {nb_label}")
        c2.metric("SVM",         f"{EMOJI_MAP[svm_label]}  {svm_label}")
        c3.metric("BERT",        f"{EMOJI_MAP[bert_label]}  {bert_label}")

        st.markdown("")
        labels  = [nb_label, svm_label, bert_label]
        counter = Counter(labels)

        if len(set(labels)) == 1:
            st.markdown(
                "<div class='verdict agree'>"
                "<span class='verdict-icon'>✓</span>"
                "<span>All three models agree — strong consensus.</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        elif len(set(labels)) == 2:
            majority = counter.most_common(1)[0][0]
            st.markdown(
                f"<div class='verdict majority'>"
                f"<span class='verdict-icon'>⊘</span>"
                f"<span>Majority verdict: <strong>{majority}</strong> — one model differs.</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='verdict split'>"
                "<span class='verdict-icon'>⚠</span>"
                "<span>All three models disagree — review may be ambiguous.</span>"
                "</div>",
                unsafe_allow_html=True,
            )
