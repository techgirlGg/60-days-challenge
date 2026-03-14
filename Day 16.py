import numpy as np

# Example embedding vectors
embedding1 = np.array([0.2, 0.5, 0.1, 0.7])
embedding2 = np.array([0.3, 0.6, 0.2, 0.8])

# Dot Product
dot_product = np.dot(embedding1, embedding2)

print("Dot Product:", dot_product)


# Matrix of multiple embeddings
embeddings = np.array([
    [0.2, 0.5, 0.1, 0.7],
    [0.3, 0.6, 0.2, 0.8],
    [0.9, 0.1, 0.4, 0.3]
])

# Matrix multiplication
result = embeddings @ embeddings.T

print("\nMatrix Multiplication Result:")
print(result)