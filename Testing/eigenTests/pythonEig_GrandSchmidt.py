import numpy as np

def gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    L = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            L[i, j] = Q[:, i] @ A[:, j]
            v = v - L[i, j] * Q[:, i]

        L[j, j] = np.linalg.norm(v)
        Q[:, j] = v / L[j, j]

    return Q, L

# Define the matrix A
A = np.array([
    [4, 1, 2, 3],
    [1, 3, 1, 2],
    [2, 1, 6, 1],
    [3, 2, 1, 5]
])

def lq_algorithm(A, num_iterations):
    for iter in range(num_iterations):
        Q, L = gram_schmidt(A)
        A = L @ Q
        print("\nMatrix Q:")
        print(Q)
        print("\nMatrix L:")
        print(L)
    return Q, L

# Perform the LQ algorithm with 2 iterations
Q, L = lq_algorithm(A, 2)

# Print the results
print("After 2 iterations:")
