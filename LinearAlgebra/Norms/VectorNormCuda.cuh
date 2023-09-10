/*
#ifndef UNTITLED_VECTORNORMCUDA_H
#define UNTITLED_VECTORNORMCUDA_H

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include "VectorNorm.h"

namespace LinearAlgebra {

    */
/**
     * @class VectorNormCuda
     * @brief A class for calculating vector norms using CUDA.
     *//*

    class VectorNormCuda {
    public:
        */
/**
         * @brief Constructor for the VectorNormCuda class.
         * @param d_vector Device pointer to the vector data.
         * @param size Size of the vector.
         * @param normType Type of norm to compute.
         * @param lP_Order Order for Lp norm (default is 2).
         *//*

        VectorNormCuda(double* d_vector, int size, VectorNormType normType, unsigned short lP_Order = 2);

        */
/**
         * @brief Get the type of the norm.
         * @return The norm type.
         *//*

        VectorNormType& type();

        */
/**
         * @brief Get the computed norm value.
         * @return The norm value.
         *//*

        double value() const;

    private:
        VectorNormType _normType;
        double* _d_vector;
        double _value;
        int _size;

        double calculateL1Norm();
        double calculateL2Norm();
        double calculateLInfNorm();
        double calculateLpNorm(double order);
        
        
    };

    */
/**
 * @brief Computes the L1 norm for a vector in parallel using CUDA.
 * 
 * The L1 norm (also known as Manhattan or Taxicab norm) is the sum of the absolute values of a vector's components.
 * This kernel is designed to run on multiple threads across multiple blocks. Each thread is responsible for accessing
 * an individual component of the vector, computing its absolute value, and storing it in shared memory. After all threads
 * synchronize, a reduction is performed within each block to sum the absolute values. The final sum for each block is written
 * to the global memory in the d_partialSum array.
 * 
 * @param d_vector Device pointer to the vector data.
 * @param d_partialSum Device pointer to store the partial sum of the absolute values.
 * @param size The size of the vector.
 *//*

    __global__ void l1NormKernel(double* d_vector, double* d_partialSum, int size);

    */
/**
     * @brief Computes the L2 (Euclidean) norm for a vector in parallel using CUDA.
     * 
     * The L2 norm is the square root of the sum of the squares of a vector's components. In this kernel, each thread
     * squares its respective component and stores the result in shared memory. After a synchronization barrier,
     * a reduction is performed within each block to aggregate the squared values. The sum of squares for each block
     * is then written to the global memory in the d_partialSum array.
     * 
     * @param d_vector Device pointer to the vector data.
     * @param d_partialSum Device pointer to store the partial sum of the squares of the components.
     * @param size The size of the vector.
     *//*

    __global__ void l2NormKernel(const double* d_vector, double* d_partialSum, int size);

    */
/**
     * @brief Computes the L∞ (Chebyshev or Maximum) norm for a vector in parallel using CUDA.
     * 
     * The L∞ norm is the maximum absolute value among a vector's components. In this kernel, each thread computes the
     * absolute value of its respective component and stores it in shared memory. After a synchronization barrier,
     * a reduction is performed within each block to find the maximum absolute value. The maximum value for each block
     * is then written to the global memory in the d_partialSum array.
     * 
     * @param d_vector Device pointer to the vector data.
     * @param d_partialSum Device pointer to store the maximum absolute values for each block.
     * @param size The size of the vector.
     *//*

    __global__ void lInfNormKernel(double* d_vector, double* d_partialSum, int size);

    */
/**
     * @brief Computes the Lp norm for a vector in parallel using CUDA.
     * 
     * The Lp norm is the p-th root of the sum of the p-th power of the absolute values of a vector's components. In this kernel,
     * each thread raises its respective component's absolute value to the p-th power and stores the result in shared memory.
     * After a synchronization barrier, a reduction is performed within each block to aggregate these values. The aggregated
     * values for each block are then written to the global memory in the d_partialSum array.
     * 
     * @param d_vector Device pointer to the vector data.
     * @param d_partialSum Device pointer to store the aggregated values after raising to the p-th power for each block.
     * @param size The size of the vector.
     * @param order The value of 'p' in the Lp norm.
     *//*

    __global__ void lpNormKernel(double* d_vector, double* d_partialSum, int size, double order);
    

} // LinearAlgebra

#endif //UNTITLED_VECTORNORMCUDA_H
*/
