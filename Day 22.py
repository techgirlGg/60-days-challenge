from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
texts = [
    "I love machine learning",
    "AI is transforming the world",
    "Machine learning is powerful",
    "I dislike spam emails",
    "Spam messages are annoying",
    "I hate junk mail"
]

labels = [1,1,1,0,0,0]   # 1 = positive/AI topic, 0 = spam

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)