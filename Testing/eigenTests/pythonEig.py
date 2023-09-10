import numpy as np
from scipy.linalg import qr

# Define the matrix A
A = np.array([
    [4, 1, 2, 3],
    [1, 3, 1, 2],
    [2, 1, 6, 1],
    [3, 2, 1, 5]
])

# Number of iterations
num_iterations = 10

# Perform the QR algorithm
for iter in range(num_iterations):
    Q, R = qr(A)
    A = np.dot(R, Q)
    print(f"Iteration {iter+1}:")
    print("Q:")
    print(Q)
    print("R:")
    print(R)
    print("A:")
    print(A)
    print("\n")
    
