import numpy as np
from collections import Counter
from scipy.stats import entropy

# Example text
text = "ai is transforming the world ai is powerful ai"

# Tokenize words
words = text.split()

# Count word occurrences
word_counts = Counter(words)

# Convert counts to numpy array
counts = np.array(list(word_counts.values()))

# Probability distribution
probabilities = counts / counts.sum()

# Display results
print("Word Probability Distribution:\n")

for word, prob in zip(word_counts.keys(), probabilities):
    print(f"{word}: {prob:.3f}")

# Optional: Calculate entropy of the distribution
entropy_value = entropy(probabilities)

print("\nEntropy of distribution:", entropy_value)