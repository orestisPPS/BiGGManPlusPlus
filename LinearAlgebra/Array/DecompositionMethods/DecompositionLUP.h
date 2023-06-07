//
// Created by hal9000 on 4/26/23.
//

#ifndef UNTITLED_DECOMPOSITIONLUP_H
#define UNTITLED_DECOMPOSITIONLUP_H

#include "MatrixDecomposition.h"

namespace LinearAlgebra {
    /**
    * SolverLUP Class
    *
    * Performs the SolverLUP decomposition on a matrix. This is a numerical algorithm used to solve linear systems of
    * equations, calculate the inverse of a matrix, and compute the determinant of a matrix.
    *
    * @tparam T The type of the matrix elements
    */
    class DecompositionLUP : public MatrixDecomposition {
        
    public:
        /**
         * Constructor for SolverLUP class
         * Performs the SolverLUP decomposition on a matrix. (PA = SolverLUP)
         * SolverLUP decomposition is a variant of SolverLUP decomposition that in addition to the lower triangular matrix L and
         * the upper triangular matrix U, also produces a permutation matrix P, which, when left-multiplied to A, 
         * reorders the rows of A. It turns out that all square matrices can be factorized in this form,[3] and the
         * factorization is numerically stable in practice.
 
         * @param matrix Pointer to the matrix to be decomposed
         * @param pivotTolerance The tolerance for pivoting the matrix
         * @param throwExceptionOnSingularMatrix (can be excluded) If true, throws an exception when a singular matrix is detected.
         *                                      If false, prints a warning message and continues.
         */
        explicit DecompositionLUP(Array<double>* matrix, double pivotTolerance = 1e-10, bool throwExceptionOnSingularMatrix = true);
        
        ~DecompositionLUP();

        /**
        * Performs the SolverLUP decomposition and creates new Array* for the L, U matrices.
        *
        * @param deleteMatrixAfterDecomposition If true, deletes the original matrix after decomposition.
        */
        void decompose (bool deleteMatrixAfterDecomposition) override;
        
        /**
         * Performs the SolverLUP and saves the L, U matrices on the original matrix.
         */
         void decomposeOnMatrix() override;
         
        /**
         * Returns a vector* with the permutation matrix P
         *
         * @return The permutation matrix P
         */
        unique_ptr<vector<unsigned>> getP();

        /**
         * Returns the invert of the matrix
         *
         * @return The inverse of the matrix
         */
         Array<double>* invertMatrix() override;

        /**
         * Returns the determinant of the matrix
         *
         * @return The determinant of the matrix
         */
         double determinant() override;

        /**
         * Returns a vector with the solution of the linear system composed by the matrix and the input RHS.
         *
         * @param rhs The right-hand side of the linear system
         *
         * @return A vector with the solution of the linear system
         */
         vector<double>* solve(vector<double>* rhs) override;

    private:
        /**
        * Permutation matrix P
        */
        vector<unsigned >* _p;
        
        /**
        * Pivot tolerance
        */
        double _pivotTolerance;
        
        /**
        * Indicates whether an exception should be thrown when a singular matrix is detected.
        */
        bool _throwExceptionOnSingularMatrix;
        /**
        * Indicates whether the matrix is singular.
        */
        bool _isSingular{};
    };

} // LinearAlgebra

#endif //UNTITLED_DECOMPOSITIONLUP_H
