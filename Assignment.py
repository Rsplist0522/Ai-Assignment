import pandas as pd
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import nltk
import re
import torch
import numpy as np
import matplotlib
import seaborn as sns
import os
import joblib 
import contractions
import emoji
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import learning_curve
from sklearn.utils import resample



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
NB_MODEL_PATH = os.path.join(MODELS_DIR, "nb_model.joblib")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.joblib")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
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

def plot_confusion_matrix(y_true, y_pred, model_name):
    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Heatmap (raw counts)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='gray')
    plt.title(f'Confusion Matrix (Counts) - {model_name}', fontsize=14)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f"confusion_matrix_{safe_name}.png")
    plt.close()
    print(f" Confusion matrix saved for {model_name}!")

    # Heatmap (normalized %)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='gray',
                vmin=0, vmax=100)
    plt.title(f'Confusion Matrix (%) - {model_name}', fontsize=14)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_norm_{safe_name}.png")
    plt.close()
    print(f" Normalised heatmap saved for {model_name}!")

# ══════════════════════════════════
#   1. Load Dataset
# ══════════════════════════════════
def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            print(f" [!] Error: File \'{file_path}\' not found.")
            return None
        data = pd.read_csv(file_path)
        if not all(col in data.columns for col in ["Rating", "Review"]):
            print(" [!] Error: Dataset missing \'Rating\' or \'Review\' columns.")
            return None
        print("Dataset loaded successfully.")
        print("=" * 50)
        print("         DATASET INFO")
        print("=" * 50)
        print(f"Total reviews : {len(data)}")
        print(f"Columns       : {data.columns.tolist()}")
        print()
        return data
    except Exception as e:
        print(f" [!] Error loading data: {e}")
        return None

# ══════════════════════════════════
#   2. Preprocessing Functions
# ══════════════════════════════════
lemmatizer = WordNetLemmatizer() #change word to base form
stop_words = set(stopwords.words("english")) #remove not important words

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) #remove numbers
    text = re.sub(r'[^a-z\s]', '', text) #remove symbol
    text = text.strip() #remove spaces
    tokens = word_tokenize(text) #let text be words
    tokens = [word for word in tokens if word not in stop_words] #remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def preprocess_bert(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = contractions.fix(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ").replace(":", "")
    text = " ".join(text.split())
    return text

def process_and_translate(text):
    DetectorFactory.seed = 0
    try:
        lang = detect(text)
        if lang != 'en':
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated, lang      
        return text, None
    except Exception:
        return text, None

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
def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    print("=" * 50)
    print(f"        TRAINING MODEL - {model_name.upper()}")
    print("=" * 50)

    model.fit(X_train, y_train)
    print(f" {model_name} trained!")

    # Learning curve for overfitting visualization
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label='Training Accuracy', marker='o')
    plt.plot(train_sizes, val_mean, label='Validation Accuracy', marker='s')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"overfitting_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.close()
    print(f" Overfitting graph saved for {model_name}!")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    plot_confusion_matrix(y_test, predictions, model_name)
    return model, {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

# BERT Specific Components
class HotelReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        # 1. Tokenize the entire list of reviews at once
        self.encodings = tokenizer(
            [preprocess_bert(r) for r in reviews],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # 2. Convert all labels to a single tensor right now
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # --- Tracking lists ---
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print(f"Training BERT on {len(train_reviews)} samples...")
    for epoch in range(4):
        # ── Training phase ──
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_function(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # ── Validation phase ──
        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss_function(outputs.logits, labels)
                total_val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = total_val_loss / len(test_loader)
        val_acc = correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f" Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # ── Plot overfitting graphs ──
    epochs = range(1, 5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label='Training Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Validation Loss', marker='s')
    ax1.set_title('BERT - Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(epochs, train_accuracies, label='Training Accuracy', marker='o')
    ax2.plot(epochs, val_accuracies, label='Validation Accuracy', marker='s')
    ax2.set_title('BERT - Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("overfitting_BERT.png")
    plt.close()
    print(" Overfitting graph saved for BERT!")

    # ── Final metrics ──
    predictions, actuals = [], []
    model.eval()
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
    plot_confusion_matrix(actual_sentiments, pred_sentiments, "BERT DistilBERT")
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
    print("Enter a review to analyze, or type \'quit\' to exit.")

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

        translated_review, detected_lang = process_and_translate(review_text)
        if detected_lang:
            print(f"Translated text   : {translated_review}")

        
        # Naive Bayes & SVM Prediction
        if 'naive_bayes' in models and 'svm' in models and 'tfidf_vectorizer' in models:
            clean_review_nb_svm = preprocess_text(translated_review)
            review_vector = models['tfidf_vectorizer'].transform([clean_review_nb_svm])
            nb_pred = models['naive_bayes'].predict(review_vector)[0]
            svm_pred = models['svm'].predict(review_vector)[0]
            print(f"\nNaïve Bayes Prediction: {nb_pred.upper()}")
            print(f"SVM Prediction          : {svm_pred.upper()}")

        # BERT Prediction
        if 'bert_model' in models and 'bert_tokenizer' in models:
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

# ══════════════════════════════════
#   Tuning
# ══════════════════════════════════
def tune_svm(X_train, y_train):
    print("=" * 50)
    print("        HYPERPARAMETER TUNING - SVM (RBF)")
    print("=" * 50)
    
    # 1. Create a smaller subset for tuning (much faster)
    X_tune, y_tune = resample(X_train, y_train, 
                              n_samples=2000, 
                              random_state=42, 
                              stratify=y_train)
    
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf']
    }
    
    grid_search = GridSearchCV(
        SVC(class_weight='balanced', random_state=42),
        param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # 2. Fit only on the tuning subset
    grid_search.fit(X_tune, y_tune)
    print(f" Best Parameters found: {grid_search.best_params_}")
    
    # 3. Train the final model with BEST parameters on FULL data
    print(" Training final SVM model on full dataset...")
    best_model = SVC(**grid_search.best_params_, class_weight='balanced', random_state=42, tol=1e-2)
    best_model.fit(X_train, y_train)
    
    return best_model


# ══════════════════════════════════
#   Main Execution Block
# ══════════════════════════════════
def main():
    models_exist = all([
        os.path.exists(NB_MODEL_PATH),
        os.path.exists(SVM_MODEL_PATH),
        os.path.exists(TFIDF_PATH),
        check_model_files(BERT_MODEL_DIR)
    ])

    if models_exist:
        print(" Found existing models. Loading from disk...")
        nb_model = joblib.load(NB_MODEL_PATH)
        svm_model = joblib.load(SVM_MODEL_PATH)
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        bert_tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_DIR)
        print(" All models loaded successfully.")
        all_results = {} 
    else:
        print(" No complete set of models found. Starting training process...")
        hotel_data = load_data("tripadvisor_hotel_reviews.csv")
        if hotel_data is None: return

        # Preprocessing
        hotel_data["Sentiment"] = hotel_data["Rating"].apply(convert_rating_to_sentiment)
        hotel_data["Clean_Review_NBSVM"] = hotel_data["Review"].apply(preprocess_text)
        hotel_data["Sentiment_Number"] = hotel_data["Sentiment"].apply(convert_sentiment_to_number)

        train_data, test_data = train_test_split(hotel_data, test_size=0.2, random_state=42, stratify=hotel_data['Sentiment'])

        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=2500)
        X_train_tfidf = tfidf_vectorizer.fit_transform(train_data["Clean_Review_NBSVM"])
        X_test_tfidf = tfidf_vectorizer.transform(test_data["Clean_Review_NBSVM"])

        # Train and Evaluate Scikit-learn Models
        nb_model, nb_results = train_evaluate_model(MultinomialNB(), X_train_tfidf, train_data["Sentiment"], X_test_tfidf, test_data["Sentiment"], "Naïve Bayes")
        best_svm_estimator = tune_svm(X_train_tfidf, train_data["Sentiment"])
        svm_model, svm_results = train_evaluate_model(best_svm_estimator, X_train_tfidf, train_data["Sentiment"], X_test_tfidf, test_data["Sentiment"], "SVM (RBF)")
        # BERT Training
        class_counts = train_data['Sentiment_Number'].value_counts().sort_index()
        # Normalizing class weights for CrossEntropyLoss
        class_weights_bert = (1 / class_counts) / (1 / class_counts).sum()
        class_weights_bert = torch.tensor(class_weights_bert.values, dtype=torch.float)

        bert_model, bert_tokenizer, bert_results = train_evaluate_bert(
            train_data['Review'].tolist(), 
            train_data['Sentiment_Number'].tolist(), 
            test_data['Review'].tolist(), 
            test_data['Sentiment_Number'].tolist(), 
            class_weights_bert
        )

        # Save everything
        print("\n Saving models for future use...")
        joblib.dump(nb_model, NB_MODEL_PATH)
        joblib.dump(svm_model, SVM_MODEL_PATH)
        joblib.dump(tfidf_vectorizer, TFIDF_PATH)
        bert_model.save_pretrained(BERT_MODEL_DIR)
        bert_tokenizer.save_pretrained(BERT_MODEL_DIR)
        print(" Models saved in \'saved_models/\' folder.")

        # Final Comparison
        all_results = {"Naïve Bayes": nb_results, "SVM": svm_results, "BERT": bert_results}
        compare_and_visualize(all_results)

    # Interactive Prediction
    trained_models = {
        'naive_bayes': nb_model,
        'svm': svm_model,
        'bert_model': bert_model,
        'bert_tokenizer': bert_tokenizer,
        'tfidf_vectorizer': tfidf_vectorizer
    }
    predict_sentiment_interactive(trained_models)

if __name__ == "__main__":
    main()