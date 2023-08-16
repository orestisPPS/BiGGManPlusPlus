//
// Created by hal9000 on 8/5/23.
//

#ifndef UNTITLED_GRAMSCHMIDTQR_H
#define UNTITLED_GRAMSCHMIDTQR_H

#include "DecompositionQR.h"



namespace LinearAlgebra {

/**
*  \class GramSchmidtQR
 * \brief This class performs QR decomposition using the Gram-Schmidt process.
 * The Gram-Schmidt process is a method for orthogonalizing a given set of vectors in an inner product space. 
 * It is widely used in numerical linear algebra to form an orthogonal basis.
 * An orthogonal basis is a set of vectors that are orthogonal (at right angles) to each other and of unit length.
 * This type of basis often simplifies computations and increases numerical stability.
 * Given a matrix A with n column vectors, the Gram-Schmidt process is as follows:
 * 1. Normalize the first vector, a1, to get the first orthonormal basis vector, e1. 
 *    This is done by dividing a1 by its Euclidean norm.
 * 2. For each subsequent vector, ai (where i = 2, 3, ..., n),
 *    subtract the projection of ai onto each of the previously computed orthonormal basis vectors, ej (where j = 1, 2, ..., i-1).
 *    This results in a vector, vi, that is orthogonal to all previously computed basis vectors.
 * 3. Normalize the orthogonal vector vi to obtain the next orthonormal basis vector, ei. This is done by dividing vi by its Euclidean norm.
 * 
 * This process is repeated for all column vectors of the matrix A, resulting in an orthonormal basis for the column space of A.
 * The Q matrix is formed by arranging the orthonormal basis vectors as columns.
 * The entries of the upper triangular matrix R are calculated by computing the dot product of the columns of A with the 
 * orthonormal vectors ei. The diagonal entries of R are the norms of the vectors vi, and the off-diagonal entries are the 
 * dot products of the vectors ai with the vectors ej, for j less than i.
 * The GramSchmidtQR class provides a single-threaded and a multi-threaded implementation for QR decomposition using the Gram-Schmidt process.
*/
    class GramSchmidtQR : public DecompositionQR {
    public:
        /**
         * @brief Constructs a GramSchmidtQR object to perform QR decomposition using the Gram-Schmidt process.
         * @param matrix Shared pointer to the matrix to decompose.
         * @param parallelizationMethod Enum indicating the method of parallelization to use.
         * @param storeOnMatrix Boolean indicating whether to store the results on the original matrix.
         */
        explicit GramSchmidtQR(bool returnQ = false, ParallelizationMethod parallelizationMethod = Wank, bool storeOnMatrix = false);

        /**
         * @brief Performs QR decomposition using the Gram-Schmidt process in a single-threaded manner.
         */
        void _singleThreadDecomposition() override;

        /**
         * @brief Performs QR decomposition using the Gram-Schmidt process in a multi-threaded manner.
         */
        void _multiThreadDecomposition() override;

        /**
         * @brief Performs QR decomposition using the Gram-Schmidt process using CUDA.
         */
        void _CUDADecomposition() override;
    };
} // LinearAlgebra

#endif //UNTITLED_GRAMSCHMIDTQR_H
