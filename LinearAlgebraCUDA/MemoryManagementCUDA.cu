
//
// Created by hal9000 on 7/26/23.
//

#include "MemoryManagementCUDA.cuh"

namespace LinearAlgebraCUDA {

    void MemoryManagementCUDA::allocateDeviceMemory(double** d_array, int size) {
        cudaError_t err = cudaMalloc((void**)d_array, size * sizeof(double));
        if (err != cudaSuccess) {
            throw std::runtime_error("Error allocating memory: " + std::string(cudaGetErrorString(err)));
        }
    }

    void MemoryManagementCUDA::copyToDevice(double* d_array, const double* h_array, int size) {
        cudaError_t err = cudaMemcpy(d_array, h_array, size * sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Error copying to device: " + std::string(cudaGetErrorString(err)));
        }
    }

    void MemoryManagementCUDA::copyToHost(double* h_array, const double* d_array, int size) {
        cudaError_t err = cudaMemcpy(h_array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Error copying to host: " + std::string(cudaGetErrorString(err)));
        }
    }

    void MemoryManagementCUDA::freeDeviceMemory(double* d_array) {
        cudaError_t err = cudaFree(d_array);
        if (err != cudaSuccess) {
            throw std::runtime_error("Error deallocating memory: " + std::string(cudaGetErrorString(err)));
        }
    }
    
}
