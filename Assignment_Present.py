import pandas as pd
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import re
import torch
import numpy as np
import os
import contractions
import emoji
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ══════════════════════════════════
#   Constants & Paths
# ══════════════════════════════════
MODELS_DIR = "saved_models"
BERT_MODEL_DIR = os.path.join(MODELS_DIR, "bert_model")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def check_model_files(directory):
    if not os.path.exists(directory):
        return False
    essential_files = ["config.json", "tokenizer_config.json"]
    has_weights = os.path.exists(os.path.join(directory, "model.safetensors")) or \
                os.path.exists(os.path.join(directory, "pytorch_model.bin"))
    has_essentials = all(os.path.exists(os.path.join(directory, f)) for f in essential_files)
    return has_essentials and has_weights

# ══════════════════════════════════
#   1. Preprocessing Functions
# ══════════════════════════════════

def preprocess_bert(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = contractions.fix(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    return text

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
#   BERT Specific Components
# ══════════════════════════════════
class HotelReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            [preprocess_bert(r) for r in reviews],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "label": self.labels[index]
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
            outputs = model(input_ids, attention_mask=attention_mask)
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
#   5. Visualization
# ══════════════════════════════════
def display_performance_table(results):
    print("=" * 50)
    print("        FINAL MODEL PERFORMANCE")
    print("=" * 50)

    comparison_df = pd.DataFrame(results).T
    comparison_df.columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    comparison_df = comparison_df * 100
    print(comparison_df.to_string(float_format="%.2f"))
    print()

# ══════════════════════════════════
#   6. User Input Function (With Validation)
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
            
        if not review_text.strip():
            print(" [!] Warning: Input is empty. Please enter a valid review.")
            continue
            
        if len(review_text.strip()) < 3:
            print(" [!] Warning: Input is too short. Please provide more detail for accurate analysis.")
            continue
            
        if not any(char.isalpha() for char in review_text):
            print(" [!] Warning: Input contains no alphabetic characters. Please enter a text review.")
            continue

        translated_review = process_and_translate(review_text)
        bert_ready_review = preprocess_bert(translated_review)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models['bert_model'].eval()
        encoded = models["bert_tokenizer"](
            bert_ready_review, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = models['bert_model'](input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(output.logits, dim=1).item()
        
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        bert_pred = sentiment_map[prediction]
        print(f"BERT Prediction         : {bert_pred.upper()}")

def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            print(f" [!] Error: File '{file_path}' not found.")
            return None
        data = pd.read_csv(file_path)
        if not all(col in data.columns for col in ["Rating", "Review"]):
            print(" [!] Error: Dataset missing 'Rating' or 'Review' columns.")
            return None
        return data
    except Exception as e:
        print(f" [!] Error loading data: {e}")
        return None

# ══════════════════════════════════
#   Main Execution Block
# ══════════════════════════════════
def main():
    if check_model_files(BERT_MODEL_DIR):
        print(" Found valid models. Loading from disk...")
        bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_DIR)
        print(" All models loaded successfully.")
    else:
        print(" No complete model found. Starting training process...")
        hotel_data = load_data("tripadvisor_hotel_reviews.csv")
        if hotel_data is None: return

        hotel_data["Sentiment"] = hotel_data["Rating"].apply(convert_rating_to_sentiment)
        hotel_data["Sentiment_Number"] = hotel_data["Sentiment"].apply(convert_sentiment_to_number)

        train_data, test_data = train_test_split(hotel_data, test_size=0.2, random_state=42, stratify=hotel_data['Sentiment'])

        class_counts = train_data['Sentiment_Number'].value_counts().sort_index()
        weights = 1.0 / class_counts 
        weights = (weights / weights.sum()) * 3.0
        class_weights_bert = torch.tensor(weights.values, dtype=torch.float)

        bert_model, bert_tokenizer, bert_results = train_evaluate_bert(
            train_data['Review'].tolist(), 
            train_data['Sentiment_Number'].tolist(), 
            test_data['Review'].tolist(), 
            test_data['Sentiment_Number'].tolist(), 
            class_weights_bert
        )

        print("\n Saving models for future use...")
        bert_model.save_pretrained(BERT_MODEL_DIR)
        bert_tokenizer.save_pretrained(BERT_MODEL_DIR)
        print(" Models saved in 'saved_models/' folder.")

        all_results = {"BERT": bert_results}
        display_performance_table(all_results)

    trained_models = {
        'bert_model': bert_model,
        'bert_tokenizer': bert_tokenizer,
    }
    predict_sentiment_interactive(trained_models)

if __name__ == "__main__":
    main()