import numpy as np
import matplotlib.pyplot as plt

def generate_matrix(n, a, b):
    if not (0 <= a < b < n):
        raise ValueError("Ensure that 0 <= a < b < n.")
    
    X = np.zeros((n, n), dtype=int)
    X[a:b+1, a:b+1] = 1
    return X

def visualize_and_save(X, filename="matrix_visualization.png"):
    plt.imshow(X, cmap='gray', origin='upper')
    plt.colorbar()
    plt.title("Matrix Visualization")
    plt.savefig(filename)
    plt.show()

# Example usage:
n = 112
a = 11
b = 26
matrix = generate_matrix(n, a, b)
visualize_and_save(matrix)
