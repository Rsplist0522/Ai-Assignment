import pandas as pd
import nltk
import re
import torch
import numpy as np
import matplotlib
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
#   NLTK Setup
# ══════════════════════════════════
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)

# ══════════════════════════════════
#   1. Data Loading with Error Handling
# ══════════════════════════════════
def load_data(file_path):
    """Loads data from a CSV file with error handling."""
    try:
        data = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully.")
        print("=" * 50)
        print("         DATASET INFO")
        print("=" * 50)
        print(f"Total reviews : {len(data)}")
        print(f"Columns       : {data.columns.tolist()}")
        print()
        return data
    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.")
        print("Please make sure the dataset is in the same directory as the script.")
        return None

# ══════════════════════════════════
#   2. Unified Preprocessing Pipeline
# ══════════════════════════════════
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """A single function to clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

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
#   Model Training & Evaluation Functions
# ══════════════════════════════════
def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Trains and evaluates a scikit-learn model."""
    print("=" * 50)
    print(f"        MODEL - {model_name.upper()}")
    print("=" * 50)
    
    model.fit(X_train, y_train)
    print("✅ Model trained!")
    
    predictions = model.predict(X_test)
    print("✅ Predictions done!")
    print()
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    print("── Results ──────────────────────────────────────")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    print()
    print("── Detailed Report ──────────────────────────────")
    print(classification_report(y_test, predictions, zero_division=0))
    
    return model, {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

# BERT Specific Components
class HotelReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=64): # Reduced max_length
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
    """Handles the training and evaluation of the DistilBERT model."""
    print("=" * 50)
    print("        MODEL - BERT (DistilBERT)")
    print("=" * 50)

    # Using DistilBERT for speed
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = HotelReviewDataset(train_reviews, train_labels, tokenizer)
    test_dataset = HotelReviewDataset(test_reviews, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    print(f"Training BERT on {len(train_reviews)} samples... (optimized for speed)")
    for epoch in range(2):
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
        print(f"✅ Epoch {epoch+1} done.")

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

    print("── Results ──────────────────────────────────────")
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    print()
    print("── Detailed Report ──────────────────────────────")
    print(classification_report(actual_sentiments, pred_sentiments, zero_division=0))

    return model, tokenizer, {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

# ══════════════════════════════════
#   5. Comparison and Visualization
# ══════════════════════════════════
def compare_and_visualize(results):
    """Prints a comparison table and saves a visualization chart."""
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
    print(f"🏆 Best Model : {best_model_name} with F1 Score of {best_f1_score:.2f}%")
    print()

    # Bar Chart
    comparison_df.plot(kind='bar', figsize=(12, 7), colormap='viridis')
    plt.title('Model Comparison', fontsize=16)
    plt.ylabel('Score (%)', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    print("✅ Chart saved as model_comparison.png!")

# ══════════════════════════════════
#   6. User Input Function
# ══════════════════════════════════
def predict_sentiment_interactive(models):
    """Allows a user to input a review and get predictions from all models."""
    print("=" * 50)
    print("        INTERACTIVE PREDICTION")
    print("=" * 50)
    print("Enter a review to analyze, or type 'quit' to exit.")

    while True:
        review_text = input("\nEnter your review: ")
        if review_text.lower() == 'quit':
            break

        clean_review = preprocess_text(review_text)

        # Naive Bayes & SVM Prediction
        review_vector = models['tfidf_vectorizer'].transform([clean_review])
        nb_pred = models['naive_bayes'].predict(review_vector)[0]
        svm_pred = models['svm'].predict(review_vector)[0]

        print(f"\nNaïve Bayes Prediction: {nb_pred.upper()}")
        print(f"SVM Prediction          : {svm_pred.upper()}")

        # BERT Prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models['bert']['model'].eval()
        encoded = models["bert_tokenizer"](
            review_text, max_length=64, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            output = models['bert']['model'](input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(output.logits, dim=1).item()
        
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        bert_pred = sentiment_map[prediction]
        print(f"BERT Prediction         : {bert_pred.upper()}")

# ══════════════════════════════════
#   Main Execution Block
# ══════════════════════════════════
def main():
    hotel_data = load_data("tripadvisor_hotel_reviews.csv")
    if hotel_data is None:
        return

    # Preprocessing and Sentiment Conversion
    hotel_data["Sentiment"] = hotel_data["Rating"].apply(convert_rating_to_sentiment)
    hotel_data["Clean_Review"] = hotel_data["Review"].apply(preprocess_text)
    hotel_data["Sentiment_Number"] = hotel_data["Sentiment"].apply(convert_sentiment_to_number)

    # Fair Train-Test Split
    train_data, test_data = train_test_split(hotel_data, test_size=0.2, random_state=42, stratify=hotel_data['Sentiment'])

    # TF-IDF Vectorization for traditional models
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data["Clean_Review"])
    X_test_tfidf = tfidf_vectorizer.transform(test_data["Clean_Review"])

    # Address Data Imbalance for scikit-learn models
    class_weights_sklearn = 'balanced'

    # Train and Evaluate Models
    nb_model, nb_results = train_evaluate_model(MultinomialNB(), X_train_tfidf, train_data["Sentiment"], X_test_tfidf, test_data["Sentiment"], "Naïve Bayes")
    svm_model, svm_results = train_evaluate_model(LinearSVC(random_state=42, max_iter=2000, class_weight=class_weights_sklearn), X_train_tfidf, train_data["Sentiment"], X_test_tfidf, test_data["Sentiment"], "SVM")

    # Address Data Imbalance for BERT
    class_counts = train_data['Sentiment_Number'].value_counts().sort_index()
    class_weights_bert = (1 / class_counts) / (1 / class_counts).sum()
    class_weights_bert = torch.tensor(class_weights_bert.values, dtype=torch.float)

    # REDUCING DATA FOR BERT (CHANGE THESE NUMBERS TO CONTROL SPEED)
    # 3000 training samples and 750 test samples is much faster
    bert_train_sample = train_data.sample(n=1000, random_state=42)
    bert_test_sample = test_data.sample(n=500, random_state=42)

    bert_model, bert_tokenizer, bert_results = train_evaluate_bert(
        bert_train_sample['Review'].tolist(), 
        bert_train_sample['Sentiment_Number'].tolist(), 
        bert_test_sample['Review'].tolist(), 
        bert_test_sample['Sentiment_Number'].tolist(), 
        class_weights_bert
    )

    # Final Comparison
    all_results = {"Naïve Bayes": nb_results, "SVM": svm_results, "BERT": bert_results}
    compare_and_visualize(all_results)

    # Interactive Prediction
    trained_models = {
        'naive_bayes': nb_model,
        'svm': svm_model,
        'bert': {'model': bert_model, 'tokenizer': bert_tokenizer},
        'tfidf_vectorizer': tfidf_vectorizer,
        'bert_tokenizer': bert_tokenizer
    }
    predict_sentiment_interactive(trained_models)

if __name__ == "__main__":
    main()
