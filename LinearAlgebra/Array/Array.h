
//
// Created by hal9000 on 1/2/23.
//


#ifndef UNTITLED_ARRAY_H
#define UNTITLED_ARRAY_H

#include <iostream>
#include <tuple>
#include <vector>
#include "cmath"
#include <limits>
#include <omp.h>
#include <memory>

using namespace std;

namespace LinearAlgebra {


/**
 * A custom matrix class that stores the matrix elements in a one-dimensional heap-allocated array. The elements
 * are stored in row-major order.
 * 
 * @tparam T The type of the elements stored in the matrix.
 */
    template<typename T>
    class Array {

    public:
        /**
         * Constructor for creating a new `Array` object.
         * 
         * @param numberOfRows The number of rows in the matrix.
         * @param numberOfColumns The number of columns in the matrix. Defaults to 1.
         * @param numberOfAisles The number of aisles in the matrix. Defaults to 1.
         * @param initialValue The initial value of all the matrix elements. Defaults to 0.
         * @param isPositiveDefinite Boolean indicating whether the matrix is symmetric and has positive eigenvalues. Defaults to false.
         */
        explicit Array(short unsigned numberOfRows, short unsigned numberOfColumns = 1, short unsigned numberOfAisles = 1,
                       T initialValue = 0, bool isPositiveDefinite = false);

        /**
         * Copy constructor for creating a new `Array` object.
         * 
         * @param array The `Array` object to be copied.
         */
        Array(const Array<T> &array);

        /**
         * The threshold number of elements after which operations on the matrix will be parallelized.
         */
        unsigned parallelizationThreshold;

        /**
        * () operator overloading. Allows to set the elements of the matrix and returns a constant reference to the element at position (i).
        *
        * @param i The row index of the element.
        * @return A reference to the element at position (i).
        * @throws std::out_of_range if the index is out of bounds.
        *@throws std::invalid_argument if the matrix is not 1D.
        */
        T& operator()(unsigned i);
        
        /**
        * () operator overloading. Allows to set the elements of the matrix and returns a reference to the element at position (i).
        *
        * @param i The row index of the element.
        * @return A reference to the element at position (i).
        * @throws std::out_of_range if the index is out of bounds.
        *@throws std::invalid_argument if the matrix is not 1D.
        */
        const T& operator()(unsigned i) const;
        
        /**
        * () operator overloading. Allows to read the elements of the matrix and returns a constant reference to the element at position (i, j).
        *
        * @param i The row index of the element to access.
        * @param j The column index of the element to access.
        * @return A constant reference to the element at position (i, j).
        * @throws std::out_of_range if the index is out of bounds.
        *@throws std::invalid_argument if the matrix is not 2D.
        */
        T& operator()(unsigned i, unsigned j);
        
        /**
        * () operator overloading. Allows to set the elements of the matrix and returns a reference to the element at position (i, j, k).
        *
        * @param i The row index of the element to access.
        * @param j The column index of the element to access.
        * @return A constant reference to the element at position (i, j).
        * @throws std::out_of_range if the index is out of bounds.
        *@throws std::invalid_argument if the matrix is not 2D.
        */
        const T& operator()(unsigned i, unsigned j) const;
        
        /**
        * () operator overloading. Allows to read the elements of the matrix and returns a constant reference to the element at position (i, j, k).
        *
        * @param i The row index of the element to access.
        * @param j The column index of the element to access.
        * @param k The aisle index of the element to access.
        * @return A constant reference to the element at position (i, j, k).
        * @throws std::out_of_range if the index is out of bounds.
        *@throws std::invalid_argument if the matrix is not 3D.
        */
        T& operator()(unsigned i, unsigned j, unsigned k);
        
        /**
        * () operator overloading. Allows to read the elements of the matrix and returns a constant reference to the element at position (i, j, k).
        *
        * @param i The row index of the element to access.
        * @param j The column index of the element to access.
        * @param k The aisle index of the element to access.
        * @return A constant reference to the element at position (i, j, k).
        * @throws std::out_of_range if the index is out of bounds.
        *@throws std::invalid_argument if the matrix is not 3D.
        */
        const T& operator()(unsigned i, unsigned j, unsigned k) const;
        
        /**
        * at() function to read and write the elements of the matrix and returns a reference to the element at position (i).
        * Throws an `out_of_range` exception if the index is out of bounds.
        * 
        * @param i The row index of the element.
        * @return A reference to the element at position (i).
        * @throws std::out_of_range if i is out of bounds.
        * @throws std::invalid_argument if the matrix is not 1D.
        */
        T& at(unsigned i);
        
        /**
        * at() function to read the elements of the matrix and returns a constant reference to the element at position (i).
        * Throws an `out_of_range` exception if the index is out of bounds.
        * 
        * @param i The row index of the element.
        * @return A constant reference to the element at position (i).
        * @throws std::out_of_range if i is out of bounds.
        * @throws std::invalid_argument if the matrix is not 1D.
         * */
        const T& at(unsigned i) const;
        
        /**
        * at() function to read and write the elements of the matrix and returns a reference to the element at position (i, j).
        * Throws an `out_of_range` exception if the index is out of bounds.
        * 
        * @param i The row index of the element to access.
        * @param j The column index of the element to access.
        * @return A reference to the element at position (i, j).
        * @throws std::out_of_range if the index is out of bounds.
        * @throws std::invalid_argument if the matrix is not 2D.
        */
        T& at(unsigned i, unsigned j);
        
        /**
        * at() function to read the elements of the matrix and returns a constant reference to the element at position (i, j).
        * Throws an `out_of_range` exception if the index is out of bounds.
        * 
        * @param i The row index of the element to access.
        * @param j The column index of the element to access.
        * @return A constant reference to the element at position (i, j).
        * @throws std::out_of_range if the index is out of bounds.
        * @throws std::invalid_argument if the matrix is not 2D.
        */
        const T& at(unsigned i, unsigned j) const;
        
        /**
        * at() function to read and write the elements of the matrix and returns a reference to the element at position (i, j, k).
        * Throws an `out_of_range` exception if the index is out of bounds.
        * 
        * @param i The row index of the element to access.
        * @param j The column index of the element to access.
        * @param k The aisle index of the element to access.
        * @return A reference to the element at position (i, j, k).
        * @throws std::out_of_range if the index is out of bounds.
        * @throws std::invalid_argument if the matrix is not 3D.
        */
        T& at(unsigned i, unsigned j, unsigned k);
        
        /**
        * at() function to read the elements of the matrix and returns a constant reference to the element at position (i, j, k).
        * Throws an `out_of_range` exception if the index is out of bounds.
        * 
        * @param i The row index of the element to access.
        * @param j The column index of the element to access.
        * @param k The aisle index of the element to access.
        * @return A constant reference to the element at position (i, j, k).
        * @throws std::out_of_range if the index is out of bounds.
        * @throws std::invalid_argument if the matrix is not 3D.
        */
        const T& at(unsigned i, unsigned j, unsigned k) const;
        
        /**
        * Overloads the assignment operator to copy the values of the provided `array` array into the current array.
        * 
        * @param array The array to copy.
        * @return A reference to the current array.
        */
        Array<T>& operator=(const Array<T>& array);
        
        /**
        * Overloads the equality operator to compare two arrays for equality.
        * 
        * @param array The array to compare with.
        * @return `true` if the arrays are equal, `false` otherwise.
        */
        bool operator==(const Array<T>& array) const;
        
        /**
        * Overloads the inequality operator to compare two arrays for inequality.
        * 
        * @param array The array to compare with.
        * @return `true` if the arrays are not equal, `false` otherwise.
        */
        bool operator!=(const Array<T>& array) const;
        
        Array<T> add(const Array<T>& array) const;
        
        void addIntoThis(const Array<T>& array);
        
        Array<T> subtract(const Array<T>& array) const;
        
        void subtractIntoThis(const Array<T>& array);
        
        Array<T> multiply(const Array<T>& array, unsigned minRow, unsigned maxRow, unsigned minCol, unsigned maxCol) const;

        Array<T> multiply(const Array<T>& array) const;
        
        vector<T> multiplyWithVector(const vector<T>& vector) const;
        
        Array<T> transpose() const;
        
        void transposeIntoThis();
        
        bool isSquare() const;
        
        bool isSymmetric() const;
        
        bool isPositiveDefinite() const;
        
        void setPositiveDefinite(bool isPositiveDefinite);
        
        bool isDiagonal() const;
        
        unsigned numberOfRows() const;

        unsigned numberOfColumns() const;

        unsigned numberOfAisles() const; 
        
        unsigned int size();

        vector<T> getRow(unsigned row);

        vector<T> getColumn(unsigned column);

        vector<T> getAisle(unsigned aisle);
        
        // Swap the elements of the i-th and j-th rows of the matrix
        void swapRows(unsigned i, unsigned j);

        // Swap the elements of the i-th and j-th columns of the matrix
        void swapColumns(unsigned i, unsigned j);
        
        void print() const;

    private:
        // The 1D array that stores the matrix. The elements are stored in row-major order.
        vector<T> _array;
        //Number of Rows. Array size : Height
        unsigned _numberOfRows;
        //Number of Columns.Array size : Width
        unsigned _numberOfColumns;
        //Number of Aisles. Array size : Depth
        unsigned _numberOfAisles;
        //Boolean that stores if the matrix is symmetric and has positive eigenvalues
        bool _isPositiveDefinite;

        bool _isSquare;

    };

} // Numerics

#endif //UNTITLED_ARRAY_H


/*




        //Performs the cholesky decomposition (A=LL^T) and returns L and L^T.
        //Applies only to symmetric positive definite matrices
        tuple<Array<double>*, Array<double>*> CholeskyDecomposition(){
            if (!_isPositiveDefinite){
                throw invalid_argument("The matrix is not square");
            }

            auto n = _numberOfRows;
            auto l = new Array<double>(_numberOfRows, _numberOfColumns);
            auto lT = new Array<double>(_numberOfRows, _numberOfColumns);
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int k = 0; k < i; k++) {
                    sum += l->at(i, k) * l->at(i, k);
                }
                l->at(i, i) = sqrt(_array[i * n + i] - sum);

                for (int j = i + 1; j < n; j++) {
                    sum = 0.0;
                    for (int k = 0; k < i; k++) {
                        sum += l->at(j, k) * l->at(i, k);
                    }
                    l->at(j, i) = (_array[j * n + i] - sum) / l->at(i, i);
                }
            }
            //Compute LT
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    lT->at(i, j) = l->at(j, i);
                }
            }
            //Return the Cholesky Decomposition of the matrix (A=LL^T)
            auto LLT = tuple<Array<double>*, Array<double>*>(l, lT);
            return LLT;
        }

        void CholeskyDecompositionOnMatrix(){
            if (!_isPositiveDefinite){
                throw std::invalid_argument("The matrix is not square");
            }
            auto n = _numberOfRows;

            // March through rows of A and L
            for (int i = 0; i < n; ++i) {
                // Compute diagonal element
                auto sum = 0.0;
                for (int k = 0; k < i; ++k) {
                    sum += _array[i * n + k] * _array[i * n + k];
                }
                _array[i * n + i] = sqrt(_array[i * n + i] - sum);

                // Compute sub-diagonal elements
                for (int j = i + 1; j < n ; ++j) {
                    sum = 0.0;
                    for (int k = 0; k < i; ++k) {
                        sum += _array[j * n + k] * _array[i * n + k];
                    }
                    _array[j * n + i] = (_array[j * n + i] - sum) / _array[i * n + i];
                }
            }

            // Store L and LT in the original object
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    _array[i * n + j] = _array[j * n + i];
                }
            }
        }*/
