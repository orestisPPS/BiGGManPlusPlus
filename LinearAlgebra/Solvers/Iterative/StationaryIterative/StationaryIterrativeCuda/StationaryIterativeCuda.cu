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
        cudaError_t err = cudaMemcpy(d_array, h_array, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            // Handle error
            printf("Error copying to device: %s\n", cudaGetErrorString(err));
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


} // LinearAlgebra
