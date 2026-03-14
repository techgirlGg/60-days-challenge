from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

texts = [
    "I love machine learning",
    "AI is transforming industries",
    "Machine learning is amazing",
    "Spam emails are annoying",
    "I hate junk mail",
    "Spam messages are irritating"
]

labels = [1,1,1,0,0,0]

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.33, stratify=labels, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))