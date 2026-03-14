import numpy as np

def compute_cosine_similarity(embeddings):
    

    # Step 1: Compute dot product matrix
    dot_product = embeddings @ embeddings.T

    # Step 2: Compute vector magnitudes
    norms = np.linalg.norm(embeddings, axis=1)

    # Step 3: Create normalization matrix
    norm_matrix = np.outer(norms, norms)

    # Step 4: Compute cosine similarity
    cosine_similarity = dot_product / norm_matrix

    return cosine_similarity


# Example embeddings (each row = text embedding)
embeddings = np.array([
    [0.2, 0.5, 0.1, 0.7],
    [0.3, 0.6, 0.2, 0.8],
    [0.9, 0.1, 0.4, 0.3]
])

# Compute similarity matrix
similarity_matrix = compute_cosine_similarity(embeddings)

print("Cosine Similarity Matrix:")
print(similarity_matrix)