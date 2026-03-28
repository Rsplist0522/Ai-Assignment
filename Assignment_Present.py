import pandas as pd
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import nltk
import re
import torch
import numpy as np
import matplotlib
import os
import joblib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ══════════════════════════════════
#   download NLTK
# ══════════════════════════════════
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)

# ══════════════════════════════════
#   Constants & Paths
# ══════════════════════════════════
MODELS_DIR = "saved_models"
BERT_MODEL_DIR = os.path.join(MODELS_DIR, "bert_model")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ══════════════════════════════════
#   1. Load Dataset
# ══════════════════════════════════
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        print("=" * 50)
        print("         DATASET INFO")
        print("=" * 50)
        print(f"Total reviews : {len(data)}")
        print(f"Columns       : {data.columns.tolist()}")
        print()
        return data
    except FileNotFoundError:
        print(f" Error: The file '{file_path}' was not found.")
        return None

# ══════════════════════════════════
#   2. Preprocessing Functions
# ══════════════════════════════════
lemmatizer = WordNetLemmatizer() #change word to base form
stop_words = set(stopwords.words("english")) #remove not important words

def preprocess_text(text): #might affect Bert hmm
    text = text.lower()
    text = re.sub(r'\d+', '', text) #remove numbers
    text = re.sub(r'[^a-z\s]', '', text) #remove symbol
    text = text.strip() #remove spaces
    tokens = word_tokenize(text) #let text be words
    tokens = [word for word in tokens if word not in stop_words] #remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def process_and_translate(text):
    DetectorFactory.seed = 0
    try:
        lang = detect(text)
        if lang != 'en':
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            print(f"Translated text: {translated}")
            return translated
        return text
    except:
        return text

# ══════════════════════════════════
#   Sentiment Conversion
# ══════════════════════════════════
def convert_rating_to_sentiment(rating):
    if rating <= 2: return "negative"
    if rating == 3: return "neutral"
    return "positive"

def convert_sentiment_to_number(sentiment):
    if sentiment == "negative": return 0
    if sentiment == "neutral": return 1
    return 2

# ══════════════════════════════════
#   Model Training & Evaluation (scikit-learn models)
# ══════════════════════════════════
# BERT Specific Components
class HotelReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = str(self.reviews[index])
        encoded = self.tokenizer(
            review,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[index], dtype=torch.long)
        }

def train_evaluate_bert(train_reviews, train_labels, test_reviews, test_labels, class_weights):
    print("=" * 50)
    print("        TRAINING MODEL - BERT (DistilBERT)")
    print("=" * 50)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = HotelReviewDataset(train_reviews, train_labels, tokenizer)
    test_dataset = HotelReviewDataset(test_reviews, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    print(f"Training BERT on {len(train_reviews)} samples...")
    for epoch in range(4):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_function(outputs.logits, labels)
            loss.backward()
            optimizer.step()
        print(f" Epoch {epoch+1} done.")

    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    pred_sentiments = [sentiment_map[p] for p in predictions]
    actual_sentiments = [sentiment_map[a] for a in actuals]

    accuracy = accuracy_score(actual_sentiments, pred_sentiments)
    precision = precision_score(actual_sentiments, pred_sentiments, average='weighted', zero_division=0)
    recall = recall_score(actual_sentiments, pred_sentiments, average='weighted', zero_division=0)
    f1 = f1_score(actual_sentiments, pred_sentiments, average='weighted', zero_division=0)

    return model, tokenizer, {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

# ══════════════════════════════════
#   5. Comparison and Visualization
# ══════════════════════════════════
def compare_and_visualize(results):
    print("=" * 50)
    print("        FINAL MODEL COMPARISON")
    print("=" * 50)

    comparison_df = pd.DataFrame(results).T
    comparison_df.columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    comparison_df = comparison_df * 100
    print(comparison_df.to_string(float_format="%.2f"))
    print()

    best_model_name = comparison_df["F1 Score"].idxmax()
    best_f1_score = comparison_df["F1 Score"].max()
    print(f" Best Model : {best_model_name} with F1 Score of {best_f1_score:.2f}%")
    print()

    comparison_df.plot(kind='bar', figsize=(12, 7), colormap='viridis')
    plt.title('Model Comparison', fontsize=16)
    plt.ylabel('Score (%)', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    print(" Chart saved as model_comparison.png!")

# ══════════════════════════════════
#   6. User Input Function
# ══════════════════════════════════
def predict_sentiment_interactive(models):
    print("=" * 50)
    print("        INTERACTIVE PREDICTION")
    print("=" * 50)
    print("Enter a review to analyze, or type 'quit' to exit.")

    while True:
        review_text = input("\nEnter your review: ")
        if review_text.lower() == 'quit':
            break
        
        # 1. Language Detection & Translation
        translated_review = process_and_translate(review_text)
        
        clean_review = preprocess_text(translated_review)

        # BERT Prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models['bert_model'].eval()
        encoded = models["bert_tokenizer"](
            translated_review, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = models['bert_model'](input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(output.logits, dim=1).item()
        
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        bert_pred = sentiment_map[prediction]
        print(f"BERT Prediction         : {bert_pred.upper()}")

# ══════════════════════════════════
#   Main Execution Block
# ══════════════════════════════════
def main():
    models_exist = all([
        os.path.exists(BERT_MODEL_DIR)
    ])

    if models_exist:
        print(" Found existing models. Loading from disk...")
        bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_DIR)
        print(" All models loaded successfully.")
    else:
        print(" No saved models found. Starting training process...")
        hotel_data = load_data("tripadvisor_hotel_reviews.csv")
        if hotel_data is None: return

        # Preprocessing
        hotel_data["Sentiment"] = hotel_data["Rating"].apply(convert_rating_to_sentiment)
        hotel_data["Clean_Review"] = hotel_data["Review"].apply(preprocess_text)
        hotel_data["Sentiment_Number"] = hotel_data["Sentiment"].apply(convert_sentiment_to_number)

        train_data, test_data = train_test_split(hotel_data, test_size=0.2, random_state=42, stratify=hotel_data['Sentiment'])

        # BERT Training
        class_counts = train_data['Sentiment_Number'].value_counts().sort_index()
        class_weights_bert = 1 / class_counts 
        class_weights_bert = torch.tensor(class_weights_bert.values, dtype=torch.float)

        # bert_train_sample = train_data.sample(n=1000, random_state=42)
        # bert_test_sample = test_data.sample(n=500, random_state=42)

        bert_model, bert_tokenizer, bert_results = train_evaluate_bert(
            train_data['Review'].tolist(), 
            train_data['Sentiment_Number'].tolist(), 
            test_data['Review'].tolist(), 
            test_data['Sentiment_Number'].tolist(), 
            class_weights_bert
        )

        # Save everything
        print("\n Saving models for future use...")
        bert_model.save_pretrained(BERT_MODEL_DIR)
        bert_tokenizer.save_pretrained(BERT_MODEL_DIR)
        print(" Models saved in 'saved_models/' folder.")

        # Final Comparison
        all_results = {"BERT": bert_results}
        compare_and_visualize(all_results)

    # Interactive Prediction
    trained_models = {
        'bert_model': bert_model,
        'bert_tokenizer': bert_tokenizer,
    }
    predict_sentiment_interactive(trained_models)

if __name__ == "__main__":
    main()
