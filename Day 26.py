from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample dataset
texts = [
    "I love machine learning",
    "AI is transforming industries",
    "Machine learning is amazing",
    "Spam emails are annoying",
    "I hate junk mail",
    "Spam messages are irritating"
]

labels = [1,1,1,0,0,0]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)

# Evaluation
print(classification_report(y_test, predictions))