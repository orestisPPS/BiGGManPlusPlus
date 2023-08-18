//
// Created by hal9000 on 7/26/23.
//

#ifndef UNTITLED_NUMERICALOPERATIONSCUDA_CUH
#define UNTITLED_NUMERICALOPERATIONSCUDA_CUH

#include <memory>
#include "MemoryManagementCUDA.cuh"

namespace LinearAlgebraCUDA {

    class NumericalOperationsCUDA {
    public:
        /**
         * \brief Computes the dot product of two vectors.
         * 
         * This method calculates the dot product (inner product) of two vectors using parallel reduction.
         * 
         * \param[in] vector1 First input vector.
         * \param[in] vector2 Second input vector.
         * \return The dot product of the two vectors.
         */
        static double dotProduct(const double* vector1, const double* vector2, int size, int threadsPerBlock = 256);

        /**
         * \brief Computes the sum of two vectors.
         * 
         * This method adds two vectors element-wise.
         * 
         * \param[in] vector1 First input vector.
         * \param[in] vector2 Second input vector.
         * \param[out] result Vector to store the resulting sum.
         * \param[in] size The size of the vectors.
         */
        static double * vectorAdd(const double* vector1, const double* vector2, int size, int threadsPerBlock = 256);

        /**
         * \brief Computes the sum of two vectors.
         * 
         * This method adds two vectors element-wise.
         * 
         * \param[in] vector1 First input vector.
         * \param[in] vector2 Second input vector.
         * \param[out] result Vector to store the resulting sum.
         * \param[in] size The size of the vectors.
         */
        static double * vectorSubtract(const double* vector1, const double* vector2, int size, int threadsPerBlock = 256);
        
        /**
         * \brief Multiplies a matrix with a vector.
         * 
         * This method computes the product of a matrix and a vector.
         * 
         * \param[in] matrix Input matrix.
         * \param[in] vector Input vector.
         * \param[out] result Vector to store the resulting product.
         * \param[in] rows Number of rows in the matrix.
         * \param[in] cols Number of columns in the matrix (size of the vector).
         */
        static double * matrixVectorMultiply(const double* matrix, const double* vector, int rows, int cols, int threadsPerBlock = 256);

        /**
         * \brief Multiplies two matrices.
         * 
         * This method computes the product of two matrices using tiling for optimized performance.
         * 
         * \param[in] matrixA First input matrix.
         * \param[in] matrixB Second input matrix.
         * \param[out] result NumericalMatrix to store the resulting product.
         * \param[in] rowsA Number of rows in matrix A.
         * \param[in] colsA Number of columns in matrix A.
         * \param[in] colsB Number of columns in matrix B.
         */
        static double * matrixMatrixMultiply(const double* matrixA, const double* matrixB, int rowsA, int colsA, int colsB, int threadsPerBlock = 256);
    };

} // LinearAlgebraCUDA

#endif //UNTITLED_NUMERICALOPERATIONSCUDA_CUH
