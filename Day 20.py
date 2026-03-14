import numpy as np

def cosine_similarity(vec1, vec2):
    # dot product
    dot_product = np.dot(vec1, vec2)
    
    # magnitudes
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # cosine similarity
    similarity = dot_product / (norm1 * norm2)
    
    return similarity


# Example embeddings
embedding1 = np.array([0.2, 0.5, 0.1, 0.7])
embedding2 = np.array([0.3, 0.6, 0.2, 0.8])

score = cosine_similarity(embedding1, embedding2)

print("Embedding Similarity:", score)