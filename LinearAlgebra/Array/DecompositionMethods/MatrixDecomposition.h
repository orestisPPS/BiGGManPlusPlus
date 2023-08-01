//
// Created by hal9000 on 4/27/23.
//

#ifndef UNTITLED_MATRIXDECOMPOSITION_H
#define UNTITLED_MATRIXDECOMPOSITION_H

#include "../Array.h"
#include <tuple>

namespace LinearAlgebra {

    class MatrixDecomposition {

    public:
        explicit MatrixDecomposition(shared_ptr<Array<double>> matrix);
        
        /**
        * Returns an Array* with the lower triangular matrix L
        *
        * @return The lower triangular matrix L
        */
        shared_ptr<Array<double>> getL();
        
        /**
        * Returns an Array* with the upper triangular matrix U
        *
        * @return The upper triangular matrix U
        */
        shared_ptr<Array<double>> getU();
        
        
        
        /**
        * Performs the SolverLUP decomposition and creates new Array* for the L, U matrices.
        *
        * @param deleteMatrixAfterDecomposition If true, deletes the original matrix after decomposition.
        */
        virtual void decompose();
        
        /**
        * Performs the SolverLUP and saves the L, U matrices on the original matrix.
        */
        virtual void decomposeOnMatrix();
        
        /**
        * Returns the invert of the matrix
        *
        * @return The inverse of the matrix
        */
        virtual shared_ptr<Array<double>> invertMatrix();
        
        /**
        * Returns the determinant of the matrix
        *
        * @return The determinant of the matrix
        */
        virtual double determinant();
        
        /**
        * Returns a vector with the solution of the linear system composed by the matrix and the input rhs.
        *
        * @param rhs The right-hand side of the linear system
        *
        * @return A vector with the solution of the linear system
        */
        virtual shared_ptr<vector<double>> solve(shared_ptr<vector<double>> rhs, shared_ptr<vector<double>> solution);


    protected:
        /**
        * Matrix to be decomposed
        */
        shared_ptr<Array<double>> _matrix;
        /**
        * Lower triangular matrix L
        */
        shared_ptr<Array<double>> _l;

        /**
        * Upper triangular matrix U
        */
        shared_ptr<Array<double>> _u;

        bool isStoredOnMatrix();

    };
} // LinearAlgebra

#endif //UNTITLED_MATRIXDECOMPOSITION_H




