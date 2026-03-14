import numpy as np

# Example embedding vectors
embeddings = np.array([
    [2.0, 3.0, 1.0, 4.0],
    [1.0, 0.5, 2.0, 3.0],
    [4.0, 2.0, 1.0, 0.5]
])

# Compute L2 norms
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

# Normalize embeddings
normalized_embeddings = embeddings / norms

print("Original Embeddings:\n", embeddings)
print("\nNormalized Embeddings:\n", normalized_embeddings)