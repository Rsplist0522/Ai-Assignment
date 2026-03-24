import pandas as pd 
import nltk 
import re 
import torch  
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ══════════════════════════════════
#   NLTK Setup
# ══════════════════════════════════
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ══════════════════════════════════
#   Load Dataset
# ══════════════════════════════════
hotel_data = pd.read_csv("tripadvisor_hotel_reviews.csv")

print("=" * 50)
print("         DATASET INFO")
print("=" * 50)
print("Total reviews :", len(hotel_data))
print("Columns       :", hotel_data.columns.tolist())
print()

# ══════════════════════════════════
#   Convert Ratings to Sentiment
# ══════════════════════════════════
def convert_rating_to_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

hotel_data["Sentiment"] = hotel_data["Rating"].apply(convert_rating_to_sentiment) #

print("Sentiment distribution:")
print(hotel_data["Sentiment"].value_counts())
print()

# ══════════════════════════════════════════════════════════
#   Preprocessing Start
# ══════════════════════════════════════════════════════════
print("=" * 50)
print("         PREPROCESSING")
print("=" * 50)

# ══════════════════════════════════════════════════════════
#   Step 1: Text Cleaning
# ══════════════════════════════════════════════════════════
def clean_review(review):
    review = review.lower()                     # make all letters lowercase
    review = re.sub(r'\d+', '', review)         # remove numbers
    review = re.sub(r'[^a-z\s]', '', review)    # remove punctuation
    review = review.strip()                     # remove extra spaces
    return review

hotel_data["Clean_Review"] = hotel_data["Review"].apply(clean_review)
print("✅ Step 1 done: Text cleaning")

# ══════════════════════════════════════════════════════════
#   Step 2: Tokenization
# ══════════════════════════════════════════════════════════
# Split sentences into individual words
def split_into_words(review):
    word_list = word_tokenize(review)
    return word_list

hotel_data["Word_List"] = hotel_data["Clean_Review"].apply(split_into_words)
print("✅ Step 2 done: Tokenization")

print()
print("── Tokenization Example ─────────────────────────")
print("Original :", hotel_data["Review"][0][:80], "...")
print("Tokens   :", hotel_data["Word_List"][0][:15], "...")
print()

# ══════════════════════════════════════════════════════════
#   Step 3: Remove Stopwords
# ══════════════════════════════════════════════════════════
# Remove common words like "the", "is", "at" that add no meaning
common_words = set(stopwords.words("english"))

def remove_common_words(word_list):
    important_words = [word for word in word_list if word not in common_words]
    return important_words

hotel_data["Word_List"] = hotel_data["Word_List"].apply(remove_common_words)
print("✅ Step 3 done: Stopword removal")

# ══════════════════════════════════════════════════════════
#   Step 4: Lemmatization
# ══════════════════════════════════════════════════════════
# Reduce words to their base form e.g. "rooms" → "room"
word_reducer = WordNetLemmatizer()

def reduce_to_root_word(word_list):
    root_words = [word_reducer.lemmatize(word) for word in word_list]
    return root_words

hotel_data["Word_List"] = hotel_data["Word_List"].apply(reduce_to_root_word)

# Join word list back into a single sentence for TF-IDF
hotel_data["Clean_Review"] = hotel_data["Word_List"].apply(lambda words: " ".join(words))
print("✅ Step 4 done: Lemmatization")

print()
print("── Before & After Preprocessing ─────────────────")
print("BEFORE:", hotel_data["Review"][0][:100], "...")
print("AFTER :", hotel_data["Clean_Review"][0][:100], "...")
print()

# ══════════════════════════════════════════════════════════
#   Step 5: TF-IDF
# ══════════════════════════════════════════════════════════
# Convert text into numbers so model can read it
tfidf_converter = TfidfVectorizer(max_features=5000)
review_features = tfidf_converter.fit_transform(hotel_data["Clean_Review"])
sentiment_labels = hotel_data["Sentiment"]

print("✅ Step 5 done: TF-IDF feature extraction")
print("Matrix shape:", review_features.shape)
print("Meaning     :", review_features.shape[0], "reviews,", review_features.shape[1], "word features")
print()

# ── Split into training and testing data ──────────────────
# 80% used to teach the model, 20% used to test the model
train_features, test_features, train_labels, test_labels = train_test_split(
    review_features, sentiment_labels, test_size=0.2, random_state=42
)

print("Training samples:", train_features.shape[0])
print("Testing samples :", test_features.shape[0])
print()

# ── Sample reviews to test all models ─────────────────────
sample_reviews = [
    "The room was clean and the staff was very friendly",
    "Terrible experience, dirty room and rude staff",
    "The hotel was okay, nothing special"
]

# ══════════════════════════════════════════════════════════
#   MODEL 1 — NAÏVE BAYES
# ══════════════════════════════════════════════════════════
print("=" * 50)
print("        MODEL 1 — NAÏVE BAYES")
print("=" * 50)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(train_features, train_labels)
print("✅ Model trained!")

naive_bayes_predictions = naive_bayes_model.predict(test_features)
print("✅ Predictions done!")
print()

print("── Results ──────────────────────────────────────")
print("Accuracy :", round(accuracy_score(test_labels, naive_bayes_predictions) * 100, 2), "%")
print("Precision:", round(precision_score(test_labels, naive_bayes_predictions, average='weighted', zero_division=0) * 100, 2), "%")
print("Recall   :", round(recall_score(test_labels, naive_bayes_predictions, average='weighted') * 100, 2), "%")
print("F1 Score :", round(f1_score(test_labels, naive_bayes_predictions, average='weighted') * 100, 2), "%")
print()
print("── Detailed Report ──────────────────────────────")
print(classification_report(test_labels, naive_bayes_predictions, zero_division=0))

print("── Sample Review Test ───────────────────────────")
for review in sample_reviews:
    cleaned_review = clean_review(review)
    word_list      = split_into_words(cleaned_review)
    word_list      = remove_common_words(word_list)
    word_list      = reduce_to_root_word(word_list)
    final_review   = " ".join(word_list)
    review_vector  = tfidf_converter.transform([final_review])
    result         = naive_bayes_model.predict(review_vector)[0]
    print(f"Review     : {review}")
    print(f"Prediction : {result.upper()}")
    print()

# ══════════════════════════════════════════════════════════
#   MODEL 2 — SVM
# ══════════════════════════════════════════════════════════
print("=" * 50)
print("        MODEL 2 — SVM")
print("=" * 50)

svm_model = LinearSVC(random_state=42, max_iter=1000)
svm_model.fit(train_features, train_labels)
print("✅ Model trained!")

svm_predictions = svm_model.predict(test_features)
print("✅ Predictions done!")
print()

print("── Results ──────────────────────────────────────")
print("Accuracy :", round(accuracy_score(test_labels, svm_predictions) * 100, 2), "%")
print("Precision:", round(precision_score(test_labels, svm_predictions, average='weighted', zero_division=0) * 100, 2), "%")
print("Recall   :", round(recall_score(test_labels, svm_predictions, average='weighted') * 100, 2), "%")
print("F1 Score :", round(f1_score(test_labels, svm_predictions, average='weighted') * 100, 2), "%")
print()
print("── Detailed Report ──────────────────────────────")
print(classification_report(test_labels, svm_predictions))

print("── Sample Review Test ───────────────────────────")
for review in sample_reviews:
    cleaned_review = clean_review(review)
    word_list      = split_into_words(cleaned_review)
    word_list      = remove_common_words(word_list)
    word_list      = reduce_to_root_word(word_list)
    final_review   = " ".join(word_list)
    review_vector  = tfidf_converter.transform([final_review])
    result         = svm_model.predict(review_vector)[0]
    print(f"Review     : {review}")
    print(f"Prediction : {result.upper()}")
    print()

# ══════════════════════════════════════════════════════════
#   MODEL 3 — BERT
# ══════════════════════════════════════════════════════════
print("=" * 50)
print("        MODEL 3 — BERT")
print("=" * 50)

# ── Convert ratings to numbers for BERT ───────────────────
# BERT needs numbers not words: 0=negative, 1=neutral, 2=positive
def convert_rating_to_number(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

sentiment_number_names = ["negative", "neutral", "positive"]
hotel_data["Sentiment_Number"] = hotel_data["Rating"].apply(convert_rating_to_number)

# Use 3000 samples so BERT runs faster
bert_data = hotel_data.sample(n=3000, random_state=42).reset_index(drop=True)

bert_train_reviews, bert_test_reviews, bert_train_labels, bert_test_labels = train_test_split(
    bert_data["Review"].tolist(),
    bert_data["Sentiment_Number"].tolist(),
    test_size=0.2,
    random_state=42
)

# ── Load BERT tokenizer ───────────────────────────────────
print("Loading BERT tokenizer... (first time may take a few minutes)")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("✅ Tokenizer loaded!")

# ── Dataset class ─────────────────────────────────────────
class HotelReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews    = reviews
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.reviews[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids"      : encoded["input_ids"].squeeze(),
            "attention_mask" : encoded["attention_mask"].squeeze(),
            "label"          : torch.tensor(self.labels[index], dtype=torch.long)
        }

# ── Create training and testing datasets ──────────────────
bert_train_dataset = HotelReviewDataset(bert_train_reviews, bert_train_labels, bert_tokenizer)
bert_test_dataset  = HotelReviewDataset(bert_test_reviews,  bert_test_labels,  bert_tokenizer)
bert_train_loader  = DataLoader(bert_train_dataset, batch_size=16, shuffle=True)
bert_test_loader   = DataLoader(bert_test_dataset,  batch_size=16)

# ── Load BERT model ───────────────────────────────────────
print("Loading BERT model... (downloading ~400MB first time)")
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
bert_model = bert_model.to(device)

bert_optimizer = AdamW(bert_model.parameters(), lr=2e-5)

# ── Train BERT ────────────────────────────────────────────
print("Training BERT... (this will take several minutes)")
print()

NUMBER_OF_EPOCHS = 2
for epoch in range(NUMBER_OF_EPOCHS):
    bert_model.train()
    total_loss     = 0
    correct_count  = 0
    total_count    = 0

    for batch_number, batch in enumerate(bert_train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        bert_optimizer.zero_grad()
        output     = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss       = output.loss
        loss.backward()
        bert_optimizer.step()

        total_loss    += loss.item()
        predictions    = torch.argmax(output.logits, dim=1)
        correct_count += (predictions == labels).sum().item()
        total_count   += labels.size(0)

        if (batch_number + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_number+1}/{len(bert_train_loader)} | Loss: {loss.item():.4f}")

    epoch_accuracy = correct_count / total_count * 100
    average_loss   = total_loss / len(bert_train_loader)
    print(f"✅ Epoch {epoch+1} done | Avg Loss: {average_loss:.4f} | Train Accuracy: {epoch_accuracy:.2f}%")
    print()

# ── Test BERT ─────────────────────────────────────────────
print("Testing BERT...")
bert_model.eval()
bert_predicted_labels = []
bert_actual_labels    = []

with torch.no_grad():
    for batch in bert_test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        output         = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions    = torch.argmax(output.logits, dim=1)
        bert_predicted_labels.extend(predictions.cpu().numpy())
        bert_actual_labels.extend(labels.cpu().numpy())

# Convert numbers back to words for the report
bert_predicted_words = [sentiment_number_names[p] for p in bert_predicted_labels]
bert_actual_words    = [sentiment_number_names[l] for l in bert_actual_labels]

print()
print("── Results ──────────────────────────────────────")
print("Accuracy :", round(accuracy_score(bert_actual_words, bert_predicted_words) * 100, 2), "%")
print("Precision:", round(precision_score(bert_actual_words, bert_predicted_words, average='weighted', zero_division=0) * 100, 2), "%")
print("Recall   :", round(recall_score(bert_actual_words, bert_predicted_words, average='weighted') * 100, 2), "%")
print("F1 Score :", round(f1_score(bert_actual_words, bert_predicted_words, average='weighted') * 100, 2), "%")
print()
print("── Detailed Report ──────────────────────────────")
print(classification_report(bert_actual_words, bert_predicted_words))

print("── Sample Review Test ───────────────────────────")
bert_model.eval()
for review in sample_reviews:
    encoded        = bert_tokenizer(
        review,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        output     = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(output.logits, dim=1).item()

    print(f"Review     : {review}")
    print(f"Prediction : {sentiment_number_names[prediction].upper()}")
    print()

# ══════════════════════════════════════════════════════════
#   FINAL COMPARISON
# ══════════════════════════════════════════════════════════
print("=" * 50)
print("        FINAL MODEL COMPARISON")
print("=" * 50)

comparison_table = {
    "Model"    : ["Naïve Bayes", "SVM", "BERT"],
    "Accuracy" : [
        round(accuracy_score(test_labels, naive_bayes_predictions) * 100, 2),
        round(accuracy_score(test_labels, svm_predictions) * 100, 2),
        round(accuracy_score(bert_actual_words, bert_predicted_words) * 100, 2)
    ],
    "Precision": [
        round(precision_score(test_labels, naive_bayes_predictions, average='weighted') * 100, 2),
        round(precision_score(test_labels, svm_predictions, average='weighted') * 100, 2),
        round(precision_score(bert_actual_words, bert_predicted_words, average='weighted') * 100, 2)
    ],
    "Recall"   : [
        round(recall_score(test_labels, naive_bayes_predictions, average='weighted') * 100, 2),
        round(recall_score(test_labels, svm_predictions, average='weighted') * 100, 2),
        round(recall_score(bert_actual_words, bert_predicted_words, average='weighted') * 100, 2)
    ],
    "F1 Score" : [
        round(f1_score(test_labels, naive_bayes_predictions, average='weighted') * 100, 2),
        round(f1_score(test_labels, svm_predictions, average='weighted') * 100, 2),
        round(f1_score(bert_actual_words, bert_predicted_words, average='weighted') * 100, 2)
    ]
}

comparison_dataframe = pd.DataFrame(comparison_table)
print(comparison_dataframe.to_string(index=False))
print()

best_model_name = comparison_dataframe.loc[comparison_dataframe["F1 Score"].idxmax(), "Model"]
best_f1_score   = comparison_dataframe["F1 Score"].max()
print(f"🏆 Best Model : {best_model_name} with F1 Score of {best_f1_score}%")
print()

# ── Bar Chart ─────────────────────────────────────────────
chart_metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
model_names   = ["Naïve Bayes", "SVM", "BERT"]
bar_colors    = ["#4e79a7", "#f28e2b", "#e15759"]

bar_positions = np.arange(len(chart_metrics))
bar_width     = 0.25

figure, chart = plt.subplots(figsize=(12, 6))

for i, (model_name, color) in enumerate(zip(model_names, bar_colors)):
    score_values = [
        comparison_dataframe.loc[comparison_dataframe["Model"] == model_name, metric].values[0]
        for metric in chart_metrics
    ]
    bars = chart.bar(bar_positions + i * bar_width, score_values, bar_width, label=model_name, color=color)

    for bar in bars:
        chart.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height()}%",
            ha="center", va="bottom", fontsize=8
        )

chart.set_xlabel("Metrics", fontsize=12)
chart.set_ylabel("Score (%)", fontsize=12)
chart.set_title("Model Comparison — Naïve Bayes vs SVM vs BERT", fontsize=14)
chart.set_xticks(bar_positions + bar_width)
chart.set_xticklabels(chart_metrics)
chart.set_ylim(0, 105)
chart.legend()
chart.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("model_comparison.png")
print("✅ Chart saved as model_comparison.png!")