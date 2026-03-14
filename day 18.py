import numpy as np
from scipy.optimize import minimize

# Example text embeddings
embedding_A = np.array([0.2, 0.5, 0.1, 0.7])
embedding_B = np.array([0.3, 0.6, 0.2, 0.8])

# Cost function
def cost_function(theta):
    return np.sum((embedding_A - theta * embedding_B) ** 2)

# Initial guess
initial_theta = 1.0

# Optimization
result = minimize(cost_function, initial_theta)

# Results
print("Optimal theta:", result.x[0])
print("Minimum cost:", result.fun)