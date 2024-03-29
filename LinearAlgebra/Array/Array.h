
//
// Created by hal9000 on 1/2/23.
//


#ifndef UNTITLED_ARRAY_H
#define UNTITLED_ARRAY_H

#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>
#include <limits>
#include <omp.h>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <iomanip>

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
                       T initialValue = 0, bool isPositiveDefinite = false) :
                _numberOfRows(numberOfRows), _numberOfColumns(numberOfColumns), _numberOfAisles(numberOfAisles),
                //_array(make_shared<vector<T>>(numberOfRows * numberOfColumns * numberOfAisles, initialValue)),
                _array(new vector<T>(numberOfRows * numberOfColumns * numberOfAisles, initialValue)),
                _isPositiveDefinite(isPositiveDefinite), _isSquare(false),
                parallelizationThreshold(1E4) {
            if (numberOfRows == numberOfColumns and numberOfColumns == numberOfAisles)
                _isSquare = true;
        }
        
        /**
         * Copy constructor for creating a new `Array` object.
         * 
         * @param array The `Array` object to be copied.
         */
        Array(const Array<T> &array) :
                _numberOfRows(array._numberOfRows), _numberOfColumns(array._numberOfColumns),
                _numberOfAisles(array._numberOfAisles),
                _array(array._array), _isPositiveDefinite(array._isPositiveDefinite), _isSquare(array._isSquare),
                parallelizationThreshold(array.parallelizationThreshold) {}
                
        ~Array() {
            _array->clear();
            delete _array;
            _array = nullptr;
        }

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
        T& operator()(unsigned i) {
            if (i >= _numberOfRows * _numberOfColumns * _numberOfAisles)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfAisles == 1 && _numberOfColumns == 1)
                return _array->at(i);
            else
                throw invalid_argument("The matrix is not one-dimensional.");
        }

        /**
        * () operator overloading. Allows to set the elements of the matrix and returns a reference to the element at position (i).
        *
        * @param i The row index of the element.
        * @return A reference to the element at position (i).
        * @throws std::out_of_range if the index is out of bounds.
        *@throws std::invalid_argument if the matrix is not 1D.
        */
        const T& operator()(unsigned i) const {
            if (i >= _numberOfRows * _numberOfColumns * _numberOfAisles)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfAisles == 1 and _numberOfColumns == 1)
                return _array->at(i);
            else
                throw invalid_argument("The matrix is not one-dimensional.");
        }

        /**
        * () operator overloading. Allows to read the elements of the matrix and returns a constant reference to the element at position (i, j).
        *
        * @param i The row index of the element to access.
        * @param j The column index of the element to access.
        * @return A constant reference to the element at position (i, j).
        * @throws std::out_of_range if the index is out of bounds.
        *@throws std::invalid_argument if the matrix is not 2D.
        */
        T& operator()(unsigned i, unsigned j) {
            if (i >= _numberOfRows or j >= _numberOfColumns)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfAisles == 1)
                return _array->at(i * _numberOfColumns + j);
            else
                throw invalid_argument("The matrix is not two-dimensional.");
        }

        /**
        * () operator overloading. Allows to set the elements of the matrix and returns a reference to the element at position (i, j, k).
        *
        * @param i The row index of the element to access.
        * @param j The column index of the element to access.
        * @return A constant reference to the element at position (i, j).
        * @throws std::out_of_range if the index is out of bounds.
        *@throws std::invalid_argument if the matrix is not 2D.
        */
        const T& operator()(unsigned i, unsigned j) const {
            if (i >= _numberOfRows or j >= _numberOfColumns)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles == 1)
                return _array->at(i * _numberOfColumns + j);
            else
                throw invalid_argument("The matrix is not two-dimensional.");
        }

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
        T& operator()(unsigned i, unsigned j, unsigned k){
            if (i >= _numberOfRows or j >= _numberOfColumns or k >= _numberOfAisles)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles > 1)
                return _array->at(i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k);
            else
                throw invalid_argument("The matrix is not three-dimensional.");
        }

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
        const T& operator()(unsigned i, unsigned j, unsigned k) const{
            if (i >= _numberOfRows or j >= _numberOfColumns or k >= _numberOfAisles)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles > 1)
                return _array->at(i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k);
            else
                throw invalid_argument("The matrix is not three-dimensional.");
        }

        /**
        * at() function to read and write the elements of the matrix and returns a reference to the element at position (i).
        * Throws an `out_of_range` exception if the index is out of bounds.
        * 
        * @param i The row index of the element.
        * @return A reference to the element at position (i).
        * @throws std::out_of_range if i is out of bounds.
        * @throws std::invalid_argument if the matrix is not 1D.
        */
        T& at(unsigned i) {
            if (i >= _numberOfRows * _numberOfColumns * _numberOfAisles)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfAisles == 1 and _numberOfColumns == 1)
                return _array->at(i);
            else
                throw invalid_argument("The matrix is not one-dimensional.");
        }

        /**
        * at() function to read the elements of the matrix and returns a constant reference to the element at position (i).
        * Throws an `out_of_range` exception if the index is out of bounds.
        * 
        * @param i The row index of the element.
        * @return A constant reference to the element at position (i).
        * @throws std::out_of_range if i is out of bounds.
        * @throws std::invalid_argument if the matrix is not 1D.
         * */
        const T& at(unsigned i) const {
            if (i >= _numberOfRows * _numberOfColumns * _numberOfAisles)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfAisles == 1 and _numberOfColumns == 1)
                return _array->at(i);
            else
                throw invalid_argument("The matrix is not one-dimensional.");
        }

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
        T& at(unsigned i, unsigned j) {
            if (i >= _numberOfRows or j >= _numberOfColumns)
                throw out_of_range("The index is out of bounds.");
            
            if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles == 1)
                return (*_array)[i * _numberOfColumns + j];
        }

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
        const T& at(unsigned i, unsigned j) const {
            if (i >= _numberOfRows or j >= _numberOfColumns)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles == 1)
                return (*_array)[i * _numberOfColumns + j];
            else
                throw invalid_argument("The matrix is not two-dimensional.");
        }

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
        T& at(unsigned i, unsigned j, unsigned k) {
            if (i >= _numberOfRows or j >= _numberOfColumns or k >= _numberOfAisles)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles > 1)
                return _array->at(i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k);
            else
                throw invalid_argument("The matrix is not three-dimensional.");
        }

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
        const T& at(unsigned i, unsigned j, unsigned k) const{
            if (i >= _numberOfRows or j >= _numberOfColumns or k >= _numberOfAisles)
                throw out_of_range("The index is out of bounds.");
            if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles > 1)
                return _array->at(i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k);
            else
                throw invalid_argument("The matrix is not three-dimensional.");
        }

        /**
        * Overloads the assignment operator to copy the values of the provided `array` array into the current array.
        * 
        * @param array The array to copy.
        * @return A reference to the current array.
        */
        Array<T>& operator=(const Array<T>& array) {
            if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
                _numberOfAisles != array._numberOfAisles)
                throw invalid_argument("The dimensions of the arrays are not the same.");
            _isSquare = array._isSquare;
            _isPositiveDefinite = array._isPositiveDefinite;
            for (int i = 0; i < array._array->size(); ++i) {
                _array[i] = array._array[i];
            }
            return *this;
        }

        /**
        * Overloads the equality operator to compare two arrays for equality.
        * 
        * @param array The array to compare with.
        * @return `true` if the arrays are equal, `false` otherwise.
        */
        bool operator==(const Array<T>& array) const {
            if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
                _numberOfAisles != array._numberOfAisles)
                return false;
            for (int i = 0; i < array._array->size(); ++i) {
                if (_array[i] != array._array[i])
                    return false;
            }
            return true;
        }

        /**
        * Overloads the inequality operator to compare two arrays for inequality.
        * 
        * @param array The array to compare with.
        * @return `true` if the arrays are not equal, `false` otherwise.
        */
        bool operator!=(const Array<T>& array) const {
            return !(*this == array);
        }

        Array<T> add(const Array<T>& array) const{
            if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
                _numberOfAisles != array._numberOfAisles)
                throw invalid_argument("The dimensions of the arrays are not the same.");
            Array<T> result(_numberOfRows, _numberOfColumns, _numberOfAisles);
            for (int i = 0; i < _array->size(); ++i) {
                result._array[i] = _array[i] + array._array[i];
            }
            return result;
        }

        void addIntoThis(const Array<T>& array){
            if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
                _numberOfAisles != array._numberOfAisles)
                throw invalid_argument("The dimensions of the arrays are not the same.");
            for (int i = 0; i < _array->size(); ++i) {
                _array[i] += array._array[i];
            }
        }

        Array<T> subtract(const Array<T>& array) const{
            if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
                _numberOfAisles != array._numberOfAisles)
                throw invalid_argument("The dimensions of the arrays are not the same.");
            Array<T> result(_numberOfRows, _numberOfColumns, _numberOfAisles);
            for (int i = 0; i < _array->size(); ++i) {
                result._array[i] = _array[i] - array._array[i];
            }
            return result;
        }

        void subtractIntoThis(const Array<T>& array){
            if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
                _numberOfAisles != array._numberOfAisles)
                throw invalid_argument("The dimensions of the arrays are not the same.");
            for (int i = 0; i < _array->size(); ++i) {
                _array[i] -= array._array[i];
            }
        }

        Array<T> multiply(const Array<T>& array, unsigned minRow, unsigned maxRow, unsigned minCol, unsigned maxCol) const;

        Array<T> multiply(const Array<T>& array) const{
            return multiply(array, 0, _numberOfRows - 1, 0, _numberOfColumns - 1);
        }

        vector<T> multiplyWithVector(const vector<T>& vector) const {
            if (_numberOfColumns != vector.size())
                throw invalid_argument("The dimensions of the array and the vector are not the same.");

            auto result  = std::vector<T>(_numberOfRows);

            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    result[i] += _array[i * _numberOfColumns + j] * vector[j];
                }
            }
            return result;
        }
        
        void scale(T scalar){
            for (auto& element : *_array) {
                element *= scalar;
            }
        }
        

        Array<T> transpose() const{
            if (_numberOfRows != _numberOfColumns)
                throw invalid_argument("The matrix is not square.");
            
            Array<T> transpose(_numberOfColumns, _numberOfRows);
            
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = i + 1; j < _numberOfColumns; ++j) {
                    transpose._array[i * _numberOfColumns + j] = _array[j * _numberOfColumns + i];
                }
            }
        }

        void transposeIntoThis() {
            auto temp_vector = vector<T>(_array->size());
            auto temp_rows = _numberOfRows;
            auto temp_columns = _numberOfColumns;

            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    temp_vector[j * temp_rows + i] = _array[i * _numberOfColumns + j];
                }
            }

            _array = temp_vector;
            _numberOfRows = temp_columns;
            _numberOfColumns = temp_rows;
        }

        bool isSquare() const {
            return _isSquare;
        }

        bool isSymmetric(double tolerance = 1E-11) const {
            if (_numberOfRows != _numberOfColumns)
                throw invalid_argument("The matrix is not square.");
            auto result = true;
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = i + 1; j < _numberOfColumns; ++j) {
                    
                    if (abs(_array->at(i * _numberOfColumns + j) - _array->at(j * _numberOfColumns + i)) < tolerance && abs(_array->at(i * _numberOfColumns + j) - _array->at(j * _numberOfColumns + i)) > 0) {
                        cout << "i: " << i << " j: " << j <<" "<<  _array->at(i * _numberOfColumns + j) - _array->at(j * _numberOfColumns + i) << endl;
                        result = false;
                    }
                }
            }
            return result;
        }

        bool isPositiveDefinite() const {
            return _isPositiveDefinite;
        }

        void setPositiveDefinite(bool isPositiveDefinite) {
            _isPositiveDefinite = isPositiveDefinite;
        }

        bool isDiagonal() const {
            for (int i = 0; i < size(); ++i) {
                for (int j = 0; j < size(); ++j) {
                    if (i != j and _array[i * _numberOfColumns + j] != 0)
                        return false;
                }
            }
        }

        bool isDiagonallyDominant() {
            auto size = _numberOfRows;
            for (int i = 0; i < size; ++i) {
                double diagonalElement = _array[i * size + i];
                double sum = 0.0;

                for (int j = 0; j < size; ++j) {
                    if (j != i) {
                        sum += std::abs(_array[i * size + j]);
                    }
                }

                if (diagonalElement <= sum) {
                    return false;
                }
            }
            return true;
        }

        unsigned numberOfRows() const{
            return _numberOfRows;
        }

        unsigned numberOfColumns() const {
            return _numberOfColumns;
        }

        unsigned numberOfAisles() const {
            return _numberOfAisles;
        }

        unsigned int size() {
            return _array->size();
        }

        
        shared_ptr<vector<T>> getRow(unsigned row){
            auto rowVector = make_shared<vector<T>>(_numberOfColumns);
            for (int i = 0; i < _numberOfColumns; ++i) {
                (*rowVector)[i] = (*_array)[row * _numberOfColumns + i];
            }
            return rowVector;
        }

        /**
        * @brief Retrieves a part of a given row from the matrix.
        * 
        * This method extracts values from the matrix starting from the `minCol` 
        * column to the `maxCol` column for the specified `row`.
        * 
        * @param row Index of the row to retrieve (0-based index).
        * @param minCol Starting column index for retrieval (0-based index).
        * @param maxCol Ending column index for retrieval (0-based index).
        * 
        * @return A shared pointer to a vector containing the extracted values.
        * 
        * @throws out_of_range if the specified row or column indices are out of valid bounds.
        */
        shared_ptr<vector<T>> getRowPartial(unsigned row, unsigned minCol, unsigned maxCol) {
            // Boundary checks for matrix dimensions.
            if (row >= _numberOfRows || minCol >= _numberOfColumns || maxCol >= _numberOfColumns) {
                throw out_of_range("Invalid row or column indices");
            }

            // Construct a vector with the correct size.
            auto rowVector = make_shared<vector<T>>(maxCol - minCol + 1);

            // Extract the values from the matrix.
            for (unsigned i = minCol; i <= maxCol; ++i) {
                rowVector->at(i - minCol) = _array->at(row * _numberOfColumns + i);
            }
            return rowVector;
        }



        void setRow(unsigned row, shared_ptr<vector<T>> rowVector){
            for (int i = 0; i < _numberOfColumns; ++i) {
                (*_array)[row * _numberOfColumns + i] = (*rowVector)[i];
            }
        }

        /**
        * @brief Sets a part of a given row in the matrix using values from the provided vector.
        * 
        * This method replaces values in the matrix starting from the `minCol` 
        * column to the `maxCol` column for the specified `row` using values from `rowVector`.
        * 
        * @param row Index of the row to set (0-based index).
        * @param minCol Starting column index for setting values (0-based index).
        * @param maxCol Ending column index for setting values (0-based index).
        * @param rowVector A shared pointer to a vector containing values to set in the matrix.
        * 
        * @throws out_of_range if the specified row or column indices are out of valid bounds.
        * @throws invalid_argument if the size of `rowVector` doesn't match the specified column range.
        */
        void setRowPartial(unsigned row, unsigned minCol, unsigned maxCol, shared_ptr<vector<T>> rowVector) {
            // Boundary checks for matrix dimensions.
            if (row >= _numberOfRows || minCol >= _numberOfColumns || maxCol >= _numberOfColumns) {
                throw out_of_range("Invalid row or column indices");
            }

            // Check if the input vector has the correct size.
            if (rowVector->size() != (maxCol - minCol + 1)) {
                throw invalid_argument("Size of rowVector doesn't match specified column range.");
            }

            // Set the values in the matrix.
            for (unsigned i = minCol; i <= maxCol; ++i) {
                _array->at(row * _numberOfColumns + i) = rowVector->at(i - minCol);
            }
        }
        

        shared_ptr<vector<T>> getColumn(unsigned column){
            auto columnVector = make_shared<vector<T>>(_numberOfRows);
            for (int i = 0; i < _numberOfRows; ++i) {
                (*columnVector)[i] = (*_array)[i * _numberOfColumns + column];
            }
            return columnVector;
        }

        /**
        * @brief Retrieves a part of a given column from the matrix.
        * 
        * This method extracts values from the matrix starting from the `minRow` 
        * row to the `maxRow` row for the specified `column`.
        * 
        * @param column Index of the column to retrieve (0-based index).
        * @param minRow Starting row index for retrieval (0-based index).
        * @param maxRow Ending row index for retrieval (0-based index).
        * 
        * @return A shared pointer to a vector containing the extracted values.
        * 
        * @throws out_of_range if the specified column or row indices are out of valid bounds.
        */
        shared_ptr<vector<T>> getColumnPartial(unsigned column, unsigned minRow, unsigned maxRow) {
            // Boundary checks for matrix dimensions.
            if (column >= _numberOfColumns || minRow >= _numberOfRows || maxRow >= _numberOfRows) {
                throw out_of_range("Invalid row or column indices");
            }

            // Construct a vector with the correct size.
            auto columnVector = make_shared<vector<T>>(maxRow - minRow + 1);

            // Extract the values from the matrix.
            for (unsigned i = minRow; i <= maxRow; ++i) {
                columnVector->at(i - minRow) = _array->at(i * _numberOfColumns + column);
            }
            return columnVector;
        }
        
        void setColumn(unsigned column, shared_ptr<vector<T>> columnVector){
            for (int i = 0; i < _numberOfRows; ++i) {
                (*_array)[i * _numberOfColumns + column] = (*columnVector)[i];
            }
        }


        /**
        * @brief Sets a part of a given column in the matrix using values from the provided vector.
        * 
        * This method replaces values in the matrix starting from the `minRow` 
        * row to the `maxRow` row for the specified `column` using values from `columnVector`.
        * 
        * @param column Index of the column to set (0-based index).
        * @param minRow Starting row index for setting values (0-based index).
        * @param maxRow Ending row index for setting values (0-based index).
        * @param columnVector A shared pointer to a vector containing values to set in the matrix.
        * 
        * @throws out_of_range if the specified column or row indices are out of valid bounds.
        * @throws invalid_argument if the size of `columnVector` doesn't match the specified row range.
        */
        void setColumnPartial(unsigned column, unsigned minRow, unsigned maxRow, shared_ptr<vector<T>> columnVector) {
            // Boundary checks for matrix dimensions.
            if (column >= _numberOfColumns || minRow >= _numberOfRows || maxRow >= _numberOfRows) {
                throw out_of_range("Invalid row or column indices");
            }

            // Check if the input vector has the correct size.
            if (columnVector->size() != (maxRow - minRow + 1)) {
                throw invalid_argument("Size of columnVector doesn't match specified row range.");
            }

            // Set the values in the matrix.
            for (unsigned i = minRow; i <= maxRow; ++i) {
                _array->at(i * _numberOfColumns + column) = columnVector->at(i - minRow);
            }
        }
        
        
        shared_ptr<vector<T>> getAisle(unsigned aisle){
            auto aisleVector = make_shared<vector<T>>(_numberOfRows * _numberOfColumns);
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    (*aisleVector)[i * _numberOfColumns + j] = (*_array)[i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + aisle];
                }
            }
            return aisleVector;
        }
        
        void getAisle(unsigned aisle, Array<T> & aisleArray){
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    aisleArray(i, j) = (*_array)[i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + aisle];
                }
            }
        }

        // Swap the elements of the i-th and j-th rows of the matrix
        void swapRows(unsigned i, unsigned j) {
            if (i == j) return; // No need to swap if i and j are the same
            // Swap the elements of the i-th and j-th rows
            for (auto k = 0; k < _numberOfColumns; k++) {
                T temp = (*this)(i, k);
                (*this)(i, k) = (*this)(j, k);
                (*this)(j, k) = temp;
            }
        }

        // Swap the elements of the i-th and j-th columns of the matrix
        void swapColumns(unsigned i, unsigned j) {
            if (i == j) return; // No need to swap if i and j are the same
            // Swap the elements of the i-th and j-th columns
            for (auto k = 0; k < _numberOfRows; k++) {
                T temp = (*this)(k, i);
                (*this)(k, i) = (*this)(k, j);
                (*this)(k, j) = temp;
            }
        }

        shared_ptr<Array<T>> getSubMatrixPtr(unsigned minRow, unsigned maxRow, unsigned minColumn, unsigned maxColumn) {
            // Boundary checks for matrix dimensions.
            if (minRow >= _numberOfRows || maxRow >= _numberOfRows || minColumn >= _numberOfColumns || maxColumn >= _numberOfColumns) {
                throw out_of_range("Invalid row or column indices");
            }

            // Construct a vector with the correct size.
            auto subMatrix = make_shared<Array<T>>((maxRow - minRow + 1), (maxColumn - minColumn + 1));

            // Extract the values from the matrix.
            for (unsigned i = minRow; i <= maxRow; ++i) {
                for (unsigned j = minColumn; j <= maxColumn; ++j) {
                    subMatrix->at(i - minRow, j - minColumn) = _array->at(i * _numberOfColumns + j);
                }
            }
            return subMatrix;
        }

        
        Array<T> getSubMatrix(unsigned minRow, unsigned maxRow, unsigned minColumn, unsigned maxColumn) {
            // Boundary checks for matrix dimensions.
            if (minRow >= _numberOfRows || maxRow >= _numberOfRows || minColumn >= _numberOfColumns || maxColumn >= _numberOfColumns) {
                throw out_of_range("Invalid row or column indices");
            }
            auto subMatrix = Array<T>(maxRow - minRow + 1, maxColumn - minColumn + 1);
            // Extract the values from the matrix.
            for (unsigned i = minRow; i <= maxRow; ++i) {
                for (unsigned j = minColumn; j <= maxColumn; ++j) {
                    subMatrix(i - minRow, j - minColumn) = _array->at(i * _numberOfColumns + j);
                }
            }
            return subMatrix;
        }
        
        void setSubMatrix(unsigned minRow, unsigned maxRow, unsigned minColumn, unsigned maxColumn, Array<T> & subMatrix) {
            // Boundary checks for matrix dimensions.
            if (minRow >= _numberOfRows || maxRow >= _numberOfRows || minColumn >= _numberOfColumns || maxColumn >= _numberOfColumns) {
                throw out_of_range("Invalid row or column indices");
            }
            // Check if the input vector has the correct size.
            if (subMatrix.numberOfRows() != (maxRow - minRow + 1) || subMatrix.numberOfColumns() != (maxColumn - minColumn + 1)) {
                throw invalid_argument("Size of subMatrix doesn't match specified row range.");
            }
            // Set the values in the matrix.
            for (unsigned i = minRow; i <= maxRow; ++i) {
                for (unsigned j = minColumn; j <= maxColumn; ++j) {
                    (*_array)[i * _numberOfColumns + j] = subMatrix->at(i - minRow, j - minColumn);
                }
            }
        }
        
        void setSubMatrix(unsigned minRow, unsigned maxRow, unsigned minColumn, unsigned maxColumn, shared_ptr<Array<T>> subMatrix) {
            // Boundary checks for matrix dimensions.
            if (minRow >= _numberOfRows || maxRow >= _numberOfRows || minColumn >= _numberOfColumns || maxColumn >= _numberOfColumns) {
                throw out_of_range("Invalid row or column indices");
            }
            // Check if the input vector has the correct size.
            if (subMatrix->numberOfRows() != (maxRow - minRow + 1) || subMatrix->numberOfColumns() != (maxColumn - minColumn + 1)) {
                throw invalid_argument("Size of subMatrix doesn't match specified row range.");
            }
            // Set the values in the matrix.
            for (unsigned i = minRow; i <= maxRow; ++i) {
                for (unsigned j = minColumn; j <= maxColumn; ++j) {
                    (*_array)[i * _numberOfColumns + j] = subMatrix->at(i - minRow, j - minColumn);
                }
            }
        }
        

        void print(int precision = 1) const {
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    std::cout << std::scientific << std::setprecision(precision) << _array->at(i * _numberOfColumns + j) << " ";
                }
                std::cout << std::endl;
            }
        }
        
        void printRow(unsigned row) const {
            for (int i = 0; i < _numberOfColumns; ++i) {
                cout << _array[row * _numberOfColumns + i] << " ";
            }
            cout << endl;
        }
        
        void printColumn(unsigned column) const {
            for (int i = 0; i < _numberOfRows; ++i) {
                cout << _array[i * _numberOfColumns + column] << " ";
            }
            cout << endl;
        }
        
        bool hasZeroInDiagonal(double tolerance = 1E-20){
            auto size = _numberOfRows * _numberOfColumns;
            for (int i = 0; i < size; i = i + _numberOfColumns + 1) {
                if (abs(_array[i]) > tolerance)
                    return false;
            }
            return true;
        }
        
        double * getArrayPointer() const {
            return _array->data();
        }

    private:
        // The 1D array that stores the matrix. The elements are stored in row-major order.
        //shared_ptr<vector<T>> _array;
        vector<T>* _array;
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