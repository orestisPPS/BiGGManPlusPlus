//
// Created by hal9000 on 4/26/23.
//

#ifndef UNTITLED_LUP_H
#define UNTITLED_LUP_H

#include "../Array.h"
#include <tuple>

namespace LinearAlgebra {
    /**
    * LUP Class
    *
    * Performs the LUP decomposition on a matrix. This is a numerical algorithm used to solve linear systems of
    * equations, calculate the inverse of a matrix, and compute the determinant of a matrix.
    *
    * @tparam T The type of the matrix elements
    */
    class LUP {
        
    public:
        /**
         * Constructor for LUP class
         * Performs the LUP decomposition on a matrix. (PA = LU)
         * LUP decomposition is a variant of LU decomposition that in addition to the lower triangular matrix L and
         * the upper triangular matrix U, also produces a permutation matrix P, which, when left-multiplied to A, 
         * reorders the rows of A. It turns out that all square matrices can be factorized in this form,[3] and the
         * factorization is numerically stable in practice.
 
         * @param matrix Pointer to the matrix to be decomposed
         * @param pivotTolerance The tolerance for pivoting the matrix
         * @param throwExceptionOnSingularMatrix (can be excluded) If true, throws an exception when a singular matrix is detected.
         *                                      If false, prints a warning message and continues.
         */
        explicit LUP(Array<double>* matrix, double pivotTolerance = 1e-10, bool throwExceptionOnSingularMatrix = true);
        
        ~LUP();

        /**
        * Performs the LUP decomposition and creates new Array* for the L, U matrices.
        *
        * @param deleteMatrixAfterDecomposition If true, deletes the original matrix after decomposition.
        */
        void decompose(bool deleteMatrixAfterDecomposition);
        
        /**
         * Performs the LUP and saves the L, U matrices on the original matrix.
         */
         void decomposeOnMatrix();

        /**
         * Returns an Array* with the lower triangular matrix L
         *
         * @return The lower triangular matrix L
         */
         unique_ptr<Array<double>> getL();

        /**
         * Returns an Array* with the upper triangular matrix U
         *
         * @return The upper triangular matrix U
         */
        unique_ptr<Array<double>> getU();

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
         Array<double>* invertMatrix();

        /**
         * Returns the determinant of the matrix
         *
         * @return The determinant of the matrix
         */
         double determinant();

        /**
         * Returns a vector with the solution of the linear system composed by the matrix and the input RHS.
         *
         * @param rhs The right-hand side of the linear system
         *
         * @return A vector with the solution of the linear system
         */
         vector<double>* solve(vector<double>* rhs);
        
        bool isStoredOnMatrix();
        
    private:
/**
 * Matrix to be decomposed
 */        Array<double>* _matrix;
/**
 * Lower triangular matrix L
 */        Array<double>* _l;
/**
 * Upper triangular matrix U
 */        Array<double>* _u;
/**
 * Permutation matrix P
 */
        vector<unsigned >* _p;
/**
 * Pivot tolerance
 */        double _pivotTolerance;
        
        bool _runMatrixDiagnostics();
        /**
 * Indicates whether an exception should be thrown when a singular matrix is detected.
 */
        bool _throwExceptionOnSingularMatrix;
        /**
 * Indicates whether the matrix is singular.
 */
        bool _isSingular;
    };

} // LinearAlgebra

#endif //UNTITLED_LUP_H
