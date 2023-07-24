//
// Created by hal9000 on 7/22/23.
//

#include "StationaryIterativeCuda.cuh"

namespace LinearAlgebra {

    StationaryIterativeCuda::StationaryIterativeCuda(double* matrix, double* vector, double* xOld, double* xNew, double* diff, int numRows, int blockSize)
    {
        this->_numRows = numRows;
        this->_blockSize = blockSize;
        _numBlocks = getNumBlocks();
        _norm = 0;

        allocateDeviceMemoryForArray(&_d_matrix, this->_numRows * this->_numRows);
        allocateDeviceMemoryForArray(&_d_rhs, this->_numRows);
        allocateDeviceMemoryForArray(&_d_xOld, this->_numRows);
        allocateDeviceMemoryForArray(&_d_xNew, this->_numRows);
        allocateDeviceMemoryForArray(&_d_diff, this->_numRows);

        copyArrayToDevice(_d_matrix, matrix, this->_numRows * this->_numRows);
        copyArrayToDevice(_d_rhs, vector, this->_numRows);
        copyArrayToDevice(_d_xOld, xOld, this->_numRows);
        copyArrayToDevice(_d_xNew, xNew, this->_numRows);
        copyArrayToDevice(_d_diff, diff, this->_numRows);

        printf("Stationary Iterative Solver with CUDA is initialized.\n");
        printf("Number of Free DOF: %d\n", this->_numRows);
        printf("Block Size (Threads/Block): %d\n", getBlockSize());
        printf("Number of Blocks: %d\n", getNumBlocks());
    }

    StationaryIterativeCuda::~StationaryIterativeCuda() {
        cudaFree(_d_matrix);
        cudaFree(_d_rhs);
        cudaFree(_d_xOld);
        cudaFree(_d_xNew);
        cudaFree(_d_diff);
    }

    void StationaryIterativeCuda::allocateDeviceMemoryForArray(double** d_array, int size) {
        cudaError_t err = cudaMalloc((void**)d_array, size * sizeof(double));
        if (err != cudaSuccess) {
            // Handle error, possibly throw an exception or print an error message.
            printf("Error allocating memory: %s\n", cudaGetErrorString(err));
        }
    }

    void StationaryIterativeCuda::copyArrayToDevice(double* d_array, double* h_array, int size) {
        cudaError_t err;
        err = cudaMemcpy(d_array, h_array, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying to device at line %d: %s\n", __LINE__, cudaGetErrorString(err));
        }
    }

    void StationaryIterativeCuda::copyArrayToHost(double *h_array, double *d_array, int size) {
        cudaError_t err = cudaMemcpy(h_array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            // Handle error
            printf("Error copying to host: %s\n", cudaGetErrorString(err));
        }
    }

    int StationaryIterativeCuda::getBlockSize() const {
        return _blockSize;  // This value is a common choice for many GPU architectures. It can be adjusted based on profiling results.
    }

    int StationaryIterativeCuda::getNumBlocks() const {
        // Calculate the number of blocks required
        return (_numRows + getBlockSize() - 1) / getBlockSize();
        // The formula ensures that we cover all rows, especially when _numRows isn't a multiple of the block size.
        // For instance, if _numRows = 1000 and block size = 128, this formula yields 8 blocks.
    }

    __global__ void kernelJobBlockGaussSeidel(const double* matrix, const double* vector, double* xOld, double* xNew, double* diff, int numRows, int blockSize) {
        auto blockId = blockIdx.x;  // Assuming 1D grid of blocks

        // Calculate the start and end rows for this block
        unsigned startRow = blockId * blockSize;
        unsigned endRow = startRow + blockSize;
        if (endRow > numRows) {
            endRow = numRows;
        }

        // Process rows sequentially within this block
        for (unsigned row = startRow; row < endRow; ++row) {
            double sum = 0.0;

            // Before diagonal
            for (unsigned j = 0; j < row; j++) {
                sum += matrix[row * numRows + j] * xNew[j];
            }

            // After diagonal
            for (unsigned j = row + 1; j < numRows; j++) {
                sum += matrix[row * numRows + j] * xOld[j];
            }

            xNew[row] = (vector[row] - sum) / matrix[row * numRows + row];
            diff[row] = xNew[row] - xOld[row];
            xOld[row] = xNew[row];
        }
    }



    /*    __global__ void kernelComputeNorm(const double* diff, double* d_norm, int numRows) {
        extern __shared__ double sdata[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        sdata[tid] = (i < numRows) ? diff[i] * diff[i] : 0; // Initialize with square of difference
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) atomicAdd(d_norm, sdata[0]);
    }*/
    void StationaryIterativeCuda::performGaussSeidelIteration() {

        //Launch kernel with 1 
        kernelJobBlockGaussSeidel<<<getNumBlocks(), 1>>>(_d_matrix, _d_rhs, _d_xOld, _d_xNew, _d_diff, _numRows, _blockSize);
        cudaDeviceSynchronize();
    }



    void StationaryIterativeCuda::getDifferenceVector(double* diff) {
        copyArrayToHost(diff, _d_diff, _numRows);
    }
    
    void StationaryIterativeCuda::getSolutionVector(double* xNew) {
        copyArrayToHost(xNew, _d_xNew, _numRows);
    }

    double StationaryIterativeCuda::getNorm() const {
        return _norm;
    }


} // LinearAlgebra
