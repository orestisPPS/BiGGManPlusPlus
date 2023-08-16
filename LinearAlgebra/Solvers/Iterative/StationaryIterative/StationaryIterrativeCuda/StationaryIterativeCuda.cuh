/**
 * \file StationaryIterativeCuda.cuh
 * \brief Provides the StationaryIterativeCuda class and associated functions to facilitate GPU-accelerated stationary iterative methods for solving linear systems.
 * \author hal9000
 * \date 7/22/23
 */

#ifndef UNTITLED_STATIONARYITERATIVECUDA_CUH
#define UNTITLED_STATIONARYITERATIVECUDA_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include "../../../../Norms/VectorNormCuda.cuh"

namespace LinearAlgebra {

    /**
     * \brief Exception class dedicated to CUDA errors.
     */
    class CudaException : public std::runtime_error {
    public:
        explicit CudaException(const char* message) : std::runtime_error(message) {}
        explicit CudaException(const std::string& message) : std::runtime_error(message) {}
    };

/**
 * \brief CUDA kernel for implementing the Jacobi iterative method.
 * 
 * This kernel performs parallel computations associated with the Jacobi iterative method to solve linear systems.
 * 
 * \param[in] matrix A flat array representation of the matrix, stored in row-major format.
 * \param[in] vector The right-hand side vector of the linear system.
 * \param[out] result The resultant vector after applying the Jacobi iterative method for one iteration.
 * \param[in] numRows The number of rows in the matrix.
 * \param[in] numColumns The number of columns in the matrix.
 */
    __global__ void kernelJobJacobi(double* matrix, double* vector, int numRows, int numColumns);

/**
 * \brief CUDA kernel for implementing the Gauss-Seidel iterative method.
 * 
 * This kernel performs parallel computations associated with the Gauss-Seidel iterative method to solve linear systems.
 * 
 * \param[in] matrix A flat array representation of the matrix, stored in row-major format.
 * \param[in] vector The right-hand side vector of the linear system.
 * \param[out] result The resultant vector after applying the Gauss-Seidel iterative method for one iteration.
 * \param[in] numRows The number of rows in the matrix.
 * \param[in] numColumns The number of columns in the matrix.
 */
    __global__ void kernelJobBlockGaussSeidel(const double* matrix, const double* vector, double* xOld, double* xNew, double* diff, int numRows, int blockSize);

/**
 * \brief CUDA kernel for implementing the Successive Over-Relaxation (SOR) iterative method.
 * 
 * This kernel performs parallel computations associated with the SOR iterative method to solve linear systems, utilizing a relaxation factor for improved convergence.
 * 
 * \param[in] matrix A flat array representation of the matrix, stored in row-major format.
 * \param[in] vector The right-hand side vector of the linear system.
 * \param[out] result The resultant vector after applying the SOR iterative method for one iteration.
 * \param[in] numRows The number of rows in the matrix.
 * \param[in] numColumns The number of columns in the matrix.
 * \param[in] relaxationFactor The relaxation factor used in the SOR method to improve convergence.
 */
    __global__ void kernelJobSOR(double* matrix, double* vector, int numRows, int numColumns, int relaxationFactor);

    /**
     * \class StationaryIterativeCuda
     * \brief Enables GPU-accelerated stationary iterative methods for linear system solutions.
     * 
     * This class abstracts the complexities associated with GPU memory management and kernel invocations.
     * It offers methods for data transfers between host and device, memory allocation, and kernel launches.
     */
    class StationaryIterativeCuda {
        
    public:
        /**
         * \brief Constructs an object, initializing it with the matrix, vector, and initial guesses.
         * 
         * During the construction process, device memory is allocated and the provided data is transferred to the device.
         * 
         * \param[in] matrix The linear system's matrix.
         * \param[in] vector The linear system's right-hand side vector.
         * \param[in] xOld The initial guess for the solution.
         * \param[in] xNew An array to store the updated solution in each iteration.
         * \param[in] numRows The linear system's dimension (number of rows in the matrix).
         * \param[in] blockSize The number of threads in a block. Choose a block size that is a multiple of 32
         *                      (since this aligns with the warp size of NVIDIA GPUs). Common choices are 128 or 256
         *                      threads per block. This number should be sufficiently small to ensure efficient execution on
         *                      the GPU but large enough to exploit parallelism
         */
        StationaryIterativeCuda(double* matrix, double* vector, double* xOld, double* xNew, double* diff, int numRows, int blockSize);

        /**
         * \brief Destructor ensuring proper deallocation of device memory.
         */
        ~StationaryIterativeCuda();

        /**
         * \brief Allocates memory on the GPU.
         * 
         * This method is responsible for reserving contiguous memory blocks on the GPU, essential for storing matrices, vectors, and intermediate results.
         * 
         * \param[out] d_array A pointer that, post-execution, points to the allocated block's starting address on the GPU.
         * \param[in] size Specifies the number of double-precision elements for which space should be allocated.
         */
        static void allocateDeviceMemoryForArray(double** d_array, int size);

        /**
         * \brief Transfers data from the CPU to the GPU.
         * 
         * This method encapsulates the process of copying data from the host (CPU) memory to the device (GPU) memory, preparing it for computations on the GPU.
         * 
         * \param[out] d_array The destination address on the GPU.
         * \param[in] h_array The source address on the CPU.
         * \param[in] size The number of double-precision elements to be transferred.
         */
        static void copyArrayToDevice(double* d_array, double* h_array, int size);

        /**
         * \brief Transfers data from the GPU to the CPU.
         * 
         * After computations on the GPU, this method facilitates the retrieval of data, copying it back to the host memory.
         * 
         * \param[out] h_array The destination address on the CPU.
         * \param[in] d_array The source address on the GPU.
         * \param[in] size The number of double-precision elements to be retrieved.
         */
        static void copyArrayToHost(double* h_array, double* d_array, int size);


        /**
         * \brief Retrieves the size of each block (i.e., the number of threads in a block).
         * 
         * In the current implementation, the block size is fixed at 128 threads for optimal performance on many GPUs.
         * 
         * \return The number of threads in each block.
         */
        int getBlockSize() const;
        
        /**
         * \brief Calculates and retrieves the number of blocks needed based on the matrix size and block size.
         * 
         * The number of blocks is determined by dividing the total number of rows in the matrix by the block size, 
         * and rounding up. This ensures that even if the matrix size isn't a multiple of the block size, 
         * all rows are still processed.
         * 
         * For instance, if there are 1000 rows and a block size of 128, 
         * the number of blocks will be ceil(1000/128) = 8 blocks.
         * 
         * \return The number of blocks needed to cover all rows of the matrix.
         */
        int getNumBlocks() const;
        
        double getNorm() const;

        void performGaussSeidelIteration();
        
        void getDifferenceVector(double *diff);

        void getSolutionVector(double *xNew);
        
        static void printDeviceSpecs();


    private:
        double* _d_matrix;      ///< Represents the linear system's matrix in the device memory.
        double* _d_rhs;         ///< Represents the right-hand side vector of the linear system in the device memory.
        double* _d_xOld;        ///< Represents the previous iteration's solution in the device memory.
        double* _d_xNew;        ///< Represents the current iteration's solution in the device memory.
        double* _d_diff;        ///< Represents the difference between the current and previous iterations' solutions in the device memory.
        double _norm;        ///< Represents the norm of the difference vector in the device memory.
        int _numRows;           ///< Denotes the dimensionality of the square linear system.
        int _blockSize;         ///< Denotes the number of threads in a block.
        int _numBlocks;         ///< Denotes the number of blocks needed to cover all rows of the matrix.
        
    };

} // end of namespace LinearAlgebra

#endif //UNTITLED_STATIONARYITERATIVECUDA_CUH
