import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv("train.csv", header=None, skiprows=1)
df.columns = ["class_index", "title", "description"]

df['text'] = df['title'].astype(str) + " " + df['description'].astype(str)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

df['text_clean'] = df['text'].apply(clean_text)


label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
df['label'] = df['class_index'].map(label_map)


X = df['text_clean']
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)


log_reg = LogisticRegression(max_iter=200, random_state=42)
log_reg.fit(X_train_tfidf, y_train)
y_pred_lr = log_reg.predict(X_val_tfidf)

print("\n🔹 Logistic Regression Results")
print("Accuracy:", accuracy_score(y_val, y_pred_lr))
print(classification_report(y_val, y_pred_lr))


nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_val_tfidf)

print("\n🔹 Naive Bayes Results")
print("Accuracy:", accuracy_score(y_val, y_pred_nb))
print(classification_report(y_val, y_pred_nb))


cm = confusion_matrix(y_val, y_pred_lr, labels=["World", "Sports", "Business", "Sci/Tech"])

plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["World","Sports","Business","Sci/Tech"],
            yticklabels=["World","Sports","Business","Sci/Tech"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

models = ["Logistic Regression", "Naive Bayes"]
accuracies = [accuracy_score(y_val, y_pred_lr), accuracy_score(y_val, y_pred_nb)]

plt.bar(models, accuracies, color=["#1f77b4", "#ff7f0e"])
plt.ylim(0.85, 0.95)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.002, f"{acc:.3f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

def predict(text, model=log_reg):
    clean = clean_text(text)
    vect = vectorizer.transform([clean])
    return model.predict(vect)[0]

sample = "Stock markets rally after economic growth reports"
print("\nTest prediction:")
print("Text:", sample)
print("Predicted category:", predict(sample))
