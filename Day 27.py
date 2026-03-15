from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dataset
texts = [
    "I love this product",
    "This is an amazing experience",
    "The movie was fantastic",
    "I hate this service",
    "This is the worst purchase",
    "Very disappointing experience",
    "Absolutely wonderful product",
    "Terrible customer support"
]

labels = [1,1,1,0,0,0,1,0]  # 1 = Positive, 0 = Negative

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# ML Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print(classification_report(y_test, predictions))