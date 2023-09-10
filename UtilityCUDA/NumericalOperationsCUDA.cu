/*
//
// Created by hal9000 on 7/26/23.
//

#include "NumericalOperationsCUDA.cuh"

namespace LinearAlgebraCUDA {
    #define TILE_WIDTH 16

    __device__ double atomicAddDouble(double* address, double val) {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
    
    // CUDA Kernel to compute the dot product of two vectors.
    __global__ void dotProductKernel(const double* a, const double* b, double* result, int N) {
        // Shared memory declaration. This memory is accessible to all threads within the same block.
        // It's used here for performing reduction within a block.
        extern __shared__ double sdata[];

        // Compute thread ID within the block
        int tid = threadIdx.x;

        // Compute the global ID of the thread. This gives a unique ID across all threads
        // in all blocks, ensuring every thread works on a unique pair of elements from vectors a and b.
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // Each thread calculates the product of a pair of elements from a and b.
        // If the thread's global ID exceeds the size of the vectors, it initializes its product to 0.
        sdata[tid] = (i < N) ? a[i] * b[i] : 0;

        // Synchronize to ensure that all threads have written their results to shared memory.
        // Reduction within the block.
        // In each iteration of the loop, threads with ID less than s sum pairs of values.
        // This loop effectively reduces the number of parallel values by half in each iteration.
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }

            // Synchronize to ensure all threads see the same shared memory contents before the next iteration.
            __syncthreads();
        }

        // Once the reduction is complete, the result for the block is stored in sdata[0].
        // The first thread of each block (tid == 0) adds this result to the global result using atomic operations.
        // This ensures concurrency correctness when multiple blocks try to update the global result simultaneously.
        if (tid == 0) atomicAddDouble(result, sdata[0]);
    }

    __global__ void vectorAddKernel(const double* a, const double* b, double* result, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {
            result[i] = a[i] + b[i];
        }
    }

    __global__ void vectorSubtractKernel(const double* a, const double* b, double* result, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {
            result[i] = a[i] - b[i];
        }
    }

    __global__ void matrixVectorMultiplyKernel(const double* matrix, const double* vector, double* result, int rows, int cols) {
        // Shared memory declaration.
        extern __shared__ double tile[];

        int row = blockIdx.x * blockDim.x + threadIdx.x;
        double temp = 0;

        // Ensure we don't go out of bounds
        if (row >= rows) return;

        // Loop over the vector in tiles
        for (int i = 0; i < cols; i += blockDim.x) {
            // Load a segment of the vector into shared memory
            if (i + threadIdx.x < cols) {
                tile[threadIdx.x] = vector[i + threadIdx.x];
            } else {
                tile[threadIdx.x] = 0;
            }
            __syncthreads();

            // Multiply the loaded tile of the matrix with the tile of the vector
            for (int j = 0; j < blockDim.x && (i + j) < cols; j++) {
                temp += matrix[row * cols + i + j] * tile[j];
            }

            __syncthreads();
        }

        // Store the result into global memory
        result[row] = temp;
    }



    __global__ void matrixMatrixMultiplyKernel(const double* A, const double* B, double* C,
                                               int rowsA, int colsA, int colsB) {
        // Shared memory used to store tiles of A and B for the block's computation
        __shared__ double sdataA[TILE_WIDTH][TILE_WIDTH];
        __shared__ double sdataB[TILE_WIDTH][TILE_WIDTH];

        // Compute row and column indices for the thread
        int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

        // Accumulator for the result
        double cValue = 0.0;

        // Loop over all tiles of A and B required to compute the thread's assigned element in C
        for (int t = 0; t < (colsA - 1) / TILE_WIDTH + 1; ++t) {

            // Load tiles of A and B into shared memory
            if (row < rowsA && t*TILE_WIDTH + threadIdx.x < colsA) {
                sdataA[threadIdx.y][threadIdx.x] = A[row * colsA + t * TILE_WIDTH + threadIdx.x];
            } else {
                sdataA[threadIdx.y][threadIdx.x] = 0.0;
            }

            if (t*TILE_WIDTH + threadIdx.y < colsA && col < colsB) {
                sdataB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * colsB + col];
            } else {
                sdataB[threadIdx.y][threadIdx.x] = 0.0;
            }

            // Synchronize to make sure the tiles are loaded
            __syncthreads();

            // Multiply the two tiles together and accumulate the result
            for (int i = 0; i < TILE_WIDTH; ++i) {
                cValue += sdataA[threadIdx.y][i] * sdataB[i][threadIdx.x];
            }

            // Synchronize to make sure that the preceding computation is done before loading two new tiles of A and B in the next iteration
            __syncthreads();
        }

        // Write the computed value to the output matrix C
        if (row < rowsA && col < colsB) {
            C[row * colsB + col] = cValue;
        }
    }


    
    double NumericalOperationsCUDA::dotProduct(const double* vector1, const double* vector2, int size, int blockSize) {
        int blocksPerGrid = (size + blockSize - 1) / blockSize;

        double* d_vector1 = nullptr;
        double* d_vector2 = nullptr;
        double* d_result = new double(0.0);

        double h_result = 0.0;

        // Use MemoryManagementCUDA for memory operations
        MemoryManagementCUDA::allocateDeviceMemory(&d_vector1, size);
        MemoryManagementCUDA::allocateDeviceMemory(&d_vector2, size);
        MemoryManagementCUDA::allocateDeviceMemory(&d_result, 1);

        MemoryManagementCUDA::copyToDevice(d_vector1, vector1, size);
        MemoryManagementCUDA::copyToDevice(d_vector2, vector2, size);
        MemoryManagementCUDA::copyToDevice(d_result, &h_result, 1);
        
        // Launch kernel
        dotProductKernel<<<blocksPerGrid, blockSize, blockSize * sizeof(double)>>>(d_vector1, d_vector2, d_result, size);

        // Copy result back from device to host
        MemoryManagementCUDA::copyToHost(&h_result, d_result, 1);

        // Cleanup device memory
        MemoryManagementCUDA::freeDeviceMemory(d_vector1);
        MemoryManagementCUDA::freeDeviceMemory(d_vector2);
        MemoryManagementCUDA::freeDeviceMemory(d_result);

        return h_result;
    }

    double* NumericalOperationsCUDA::vectorAdd(const double* vector1, const double* vector2, int size, int threadsPerBlock) {
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        double* d_vector1 = nullptr;
        double* d_vector2 = nullptr;
        double* d_result = nullptr;

        // Use MemoryManagementCUDA for memory operations
        MemoryManagementCUDA::allocateDeviceMemory(&d_vector1, size);
        MemoryManagementCUDA::allocateDeviceMemory(&d_vector2, size);
        MemoryManagementCUDA::allocateDeviceMemory(&d_result, size);

        MemoryManagementCUDA::copyToDevice(d_vector1, vector1, size);
        MemoryManagementCUDA::copyToDevice(d_vector2, vector2, size);

        // Launch kernel
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vector1, d_vector2, d_result, size);

        double* h_result = new double[size];
        MemoryManagementCUDA::copyToHost(h_result, d_result, size);

        // Cleanup device memory
        MemoryManagementCUDA::freeDeviceMemory(d_vector1);
        MemoryManagementCUDA::freeDeviceMemory(d_vector2);
        MemoryManagementCUDA::freeDeviceMemory(d_result);

        return h_result;
    }

    double *  NumericalOperationsCUDA::vectorSubtract(const double* vector1, const double* vector2, int size, int threadsPerBlock) {
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        double* d_vector1 = nullptr;
        double* d_vector2 = nullptr;
        double* d_result = nullptr;

        // Use MemoryManagementCUDA for memory operations
        MemoryManagementCUDA::allocateDeviceMemory(&d_vector1, size);
        MemoryManagementCUDA::allocateDeviceMemory(&d_vector2, size);
        MemoryManagementCUDA::allocateDeviceMemory(&d_result, size);

        MemoryManagementCUDA::copyToDevice(d_vector1, vector1, size);
        MemoryManagementCUDA::copyToDevice(d_vector2, vector2, size);

        // Launch kernel
        vectorSubtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vector1, d_vector2, d_result, size);

        double* h_result = new double[size];
        MemoryManagementCUDA::copyToHost(h_result, d_result, size);

        // Cleanup device memory
        MemoryManagementCUDA::freeDeviceMemory(d_vector1);
        MemoryManagementCUDA::freeDeviceMemory(d_vector2);
        MemoryManagementCUDA::freeDeviceMemory(d_result);

        return h_result;
    }



    double *  NumericalOperationsCUDA::matrixVectorMultiply(const double* matrix, const double* vector, int rows, int cols, int threadsPerBlock) {
        dim3 blockSize(threadsPerBlock);
        dim3 blocksPerGrid((rows + blockSize.x - 1) / blockSize.x);

        double* d_matrix = nullptr;
        double* d_vector = nullptr;
        double* d_result = nullptr;
        double* h_result = new double[rows];

        // Use MemoryManagementCUDA for memory operations
        MemoryManagementCUDA::allocateDeviceMemory(&d_matrix, rows * cols);
        MemoryManagementCUDA::allocateDeviceMemory(&d_vector, cols);
        MemoryManagementCUDA::allocateDeviceMemory(&d_result, rows);

        MemoryManagementCUDA::copyToDevice(d_matrix, matrix, rows * cols);
        MemoryManagementCUDA::copyToDevice(d_vector, vector, cols);

        // Determine shared memory size
        int sharedMemSize = threadsPerBlock * sizeof(double);

        // Launch kernel
        matrixVectorMultiplyKernel<<<blocksPerGrid, blockSize, sharedMemSize>>>(d_matrix, d_vector, d_result, rows, cols);

        MemoryManagementCUDA::copyToHost(h_result, d_result, rows);

        // Cleanup device memory
        MemoryManagementCUDA::freeDeviceMemory(d_matrix);
        MemoryManagementCUDA::freeDeviceMemory(d_vector);
        MemoryManagementCUDA::freeDeviceMemory(d_result);

        return h_result;
    }


    double *  NumericalOperationsCUDA::matrixMatrixMultiply(const double* matrixA, const double* matrixB, int rowsA, int colsA, int colsB, int threadsPerBlock) {
        dim3 blockSize(threadsPerBlock, threadsPerBlock);  // 2D block
        dim3 blocksPerGrid((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);

        double* d_matrixA = nullptr;
        double* d_matrixB = nullptr;
        double* d_result = nullptr;

        // Use MemoryManagementCUDA for memory operations
        MemoryManagementCUDA::allocateDeviceMemory(&d_matrixA, rowsA * colsA);
        MemoryManagementCUDA::allocateDeviceMemory(&d_matrixB, colsA * colsB);
        MemoryManagementCUDA::allocateDeviceMemory(&d_result, rowsA * colsB);

        MemoryManagementCUDA::copyToDevice(d_matrixA, matrixA, rowsA * colsA);
        MemoryManagementCUDA::copyToDevice(d_matrixB, matrixB, colsA * colsB);

        // Launch kernel
        matrixMatrixMultiplyKernel<<<blocksPerGrid, blockSize>>>(d_matrixA, d_matrixB, d_result, rowsA, colsA, colsB);

        double* h_result = new double[rowsA * colsB];
        MemoryManagementCUDA::copyToHost(h_result, d_result, rowsA * colsB);

        // Cleanup device memory
        MemoryManagementCUDA::freeDeviceMemory(d_matrixA);
        MemoryManagementCUDA::freeDeviceMemory(d_matrixB);
        MemoryManagementCUDA::freeDeviceMemory(d_result);

        return h_result;
    }


} // UtilityCUDA

*/
