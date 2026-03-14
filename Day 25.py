from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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

# Create pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression())
])

# Parameter grid
param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "model__C": [0.1, 1, 10]
}

# Grid Search
grid = GridSearchCV(pipeline, param_grid, cv=3)

# Train
grid.fit(texts, labels)

# Best parameters
print("Best Parameters:", grid.best_params_)

# Best score
print("Best Accuracy:", grid.best_score_)