
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
print("Loading dataset...")

df = pd.read_csv("data/spam.csv", encoding="latin-1")

# 2. Keep only the required columns
df = df.iloc[:, :2].copy()
df.columns = ["label", "message"]

# 3. Clean the dataset
df = df.dropna(subset=["label", "message"])

df["label"] = df["label"].astype(str).str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()

# Keep only valid labels
df = df[df["label"].isin(["ham", "spam"])]

# Remove duplicate records
df = df.drop_duplicates(subset=["label", "message"])

# Convert labels to numbers
# ham = 0, spam = 1
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nDataset cleaned successfully!")
print("Total messages:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts().rename({
    0: "Not Spam",
    1: "Spam"
}))

# 4. Separate input and output
X = df["message"]
y = df["label"]

# 5. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 6. Create the machine learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000,
            strip_accents="unicode"
        )
    ),
    (
        "classifier",
        MultinomialNB(alpha=0.5)
    )
])

# 7. Train the model
print("\nTraining machine learning model...")

model.fit(X_train, y_train)

print("Training completed successfully!")

# 8. Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Spam", "Spam"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )
)

# 9. Save the trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/spam_model.pkl")

print("\nModel saved successfully!")
print("Location: models/spam_model.pkl")

# 10. Test sample messages
test_messages = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hello, are we meeting tomorrow at 10 AM?",
    "URGENT! You have won cash. Click here to claim your reward."
]

print("\n========== SAMPLE PREDICTIONS ==========")

for message in test_messages:

    prediction = model.predict([message])[0]

    result = "SPAM" if prediction == 1 else "NOT SPAM"

    print("\nMessage:", message)
    print("Prediction:", result)

print("\nProject model training finished!")