/*
//
// Created by hal9000 on 1/3/23.
//

//
// Created by hal9000 on 1/2/23.
//

#include "Array.h"
*/
/*    template <typename T>
    Array<T>::Array(unsigned rows) {
        this->_numberOfRows = rows;
        this->_numberOfColumns = 1;
        this->_numberOfAisles = 1;
        _array = new T[rows];
    }*//*


*/
/*    template<typename T>
    Array<T>::Array(unsigned rows, unsigned columns) {
        _numberOfRows = rows;
        _numberOfColumns = columns;
        _numberOfAisles = 1;
        _array = new T[_numberOfRows * _numberOfColumns];
    }*//*

*/
/*    template<typename T>
    Array<T>::Array(unsigned rows, unsigned columns, unsigned aisles) {
        _numberOfRows = rows;
        _numberOfColumns = columns;
        _numberOfAisles = aisles;
        _array = new T[_numberOfRows * _numberOfColumns * _numberOfAisles];
    }*//*



*/
/*    template<typename T>
    Array<T>::Array(const Array<T> &matrix) {
        _numberOfRows = matrix._numberOfRows;
        _numberOfColumns = matrix._numberOfColumns;
        _array = new T[_numberOfRows * _numberOfColumns];
        for (int i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            _array[i] = matrix._array[i];
        }
    }*//*


*/
/*    template<typename T>
    Array<T>::~Array() {
        delete[] _array;
        _array = nullptr;
    }*//*


*/
/*    template<typename T>
    unsigned Array<T>::numberOfRows() {
        return _numberOfRows;
    }

    template<typename T>
    unsigned Array<T>::numberOfColumns() {
        return _numberOfColumns;
    }

    template<typename T>
    unsigned Array<T>::numberOfAisles() {
        return _numberOfAisles;
    }*//*


*/
/*    template <typename T>
    unsigned Array<T>::size() {
        return _numberOfRows * _numberOfColumns * _numberOfAisles;
    }*//*


*/
/*    template<typename T>
    T Array<T>::element(unsigned row) {
        return _array[row];
    }*//*


    template<typename T>
    T Array<T>::element(unsigned row, unsigned column) {
        return _array[row * _numberOfColumns + column];
    }

    template<typename T>
    T Array<T>::element(unsigned row, unsigned column, unsigned aisle) {
        return _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle];
    }

    template<typename T>
    unsigned Array<T>::index(unsigned row) {
        return static_cast<unsigned int>(_array[row]);
    }

    template<typename T>
    unsigned Array<T>::index(unsigned row, unsigned column) {
        return _array[row * _numberOfColumns + column];
    }

    template<typename T>
    unsigned Array<T>::index(unsigned row, unsigned column, unsigned aisle) {
        return _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle];
    }

    template<typename T>
    void Array<T>::populateElement(unsigned row, T *value) {
        _array[index(row)] = value;
    }

    template<typename T>
    void Array<T>::populateElement(unsigned row, unsigned column, T *value) {
        _array[index(row, column)] = *value;
    }

    template<typename T>
    void Array<T>::populateElement(unsigned row, unsigned column, unsigned aisle, T *value) {
        _array[index(row, column, aisle)] = *value;
    }

    template<typename T>
    bool Array<T>::isSquare() {
        return _numberOfRows == _numberOfColumns;
    }

    template<typename T>
    bool Array<T>::isCubic() {
        return _numberOfRows == _numberOfColumns && _numberOfColumns == _numberOfAisles;
    }

    template <typename T>
    bool isDiagonal() {
        return false;
    }

    template<typename T>
    bool Array<T>::isVector() {
       
    }

    template<typename T>
    bool Array<T>::isSymmetric() {
        if (!isSquare() || isVector() == true || _numberOfAisles > 1) {
            return false;
        }
        for (int i = 0; i < _numberOfRows; i++) {
            for (int j = i; j < _numberOfColumns; j++) {
                if (_array[index(i, j)] != _array[index(j, i)]) {
                    return false;
                }
            }
        }
        return true;
    }

    template<typename T>
    Array<T> Array<T>::transpose() {
        if (!isSquare() || isVector() == true || _numberOfAisles > 1) {
            throw "Cannot transpose a non-square matrix, a vector, or a matrix with more than one aisle.";
        }
        auto transpose = new Array<T>(_numberOfColumns, _numberOfRows);
        for (unsigned i = 0; i < _numberOfRows; i++)
            for (unsigned j = 0; j < _numberOfColumns; j++) {
                transpose->_array[transpose->index(j, i)] = _array[index(i, j)];
            }
        return transpose;
    }

    template<typename T>
    void Array<T>::transposeIntoThis() {
        if (!isSquare() || isVector() == true || _numberOfAisles > 1) {
            throw "Cannot transpose a non-square matrix, a vector, or a matrix with more than one aisle.";
        }
        auto *transpose = new Array<T>(_numberOfColumns, _numberOfRows);
        for (unsigned i = 0; i < _numberOfRows; i++) {
            for (unsigned j = 0; j < _numberOfColumns; j++) {
                transpose->element(j, i) = element(index(i, j));
            }
        }
        _numberOfRows = transpose->_numberOfRows;
        _numberOfColumns = transpose->_numberOfColumns;
        delete[] _array;
        _array = transpose->_array;
    }

    // Overloaded assignment operator
    template<typename T>
    Array<T>& Array<T>::operator=(const Array<T> &matrix) {
        if (this != &matrix) {
            this->~Array();
            this->_numberOfRows = matrix._numberOfRows;
            this->_numberOfColumns = matrix._numberOfColumns;
            this->_numberOfAisles = matrix._numberOfAisles;
            this->_array = new T[this-> _numberOfRows * this->_numberOfColumns];
            for (unsigned i = 0; i <this-> _numberOfRows * this->_numberOfColumns; i++) {
                this->_array[i] = matrix._array[i];
            }
        }
        return *this;
    }

    // Overloaded equality operator
    template<typename T>
    bool Array<T>::operator==(const Array<T> &matrix) {
        if (_numberOfRows != matrix.numberOfRows() || _numberOfColumns != matrix.numberOfColumns() || _numberOfAisles != matrix.numberOfAisles()) {
            return false;
        }
        for (unsigned i = 0; i < size(); i++) {
            if (_array[i] != matrix._array[i]) {
                return false;
            }
        }
        return true;
    }

    // Overloaded inequality operator
    template<typename T>
    bool Array<T>::operator!=(const Array<T> &matrix) {
        if (*this == matrix) {
            return false;
        }
    }
    //-----------------------------------------
    //----- Overloaded addition operator-------
    //-----------------------------------------

    template<typename T>
    Array<T> Array<T>::operator+(const Array<T> &matrix) {
        if (_numberOfRows != matrix.numberOfRows() || _numberOfColumns != matrix.numberOfColumns() || _numberOfAisles != matrix.numberOfAisles()) {
            throw "Cannot add matrices of different dimensions.";
        }
        auto *sum = new Array<T>(_numberOfRows, _numberOfColumns, _numberOfAisles);
        for (unsigned i = 0; i < size(); i++) {
            sum->_array[i] = _array[i] + matrix._array[i];
        }
        return sum;
    }

    template<typename T>
    Array<T> Array<T>::operator-(const Array<T> &matrix) {
        if (_numberOfRows != matrix.numberOfRows() || _numberOfColumns != matrix.numberOfColumns() || _numberOfAisles != matrix.numberOfAisles()) {
            throw "Cannot add matrices of different dimensions.";
        }
        auto *sum = new Array<T>(_numberOfRows, _numberOfColumns, _numberOfAisles);
        for (unsigned i = 0; i < size(); i++) {
            sum->_array[i] = _array[i] - matrix._array[i];
        }
        return sum;
    }

    template<typename T>
    Array<T> Array<T>::operator*(const Array<T> &matrix) {
        if (_numberOfColumns != matrix.numberOfRows()) {
            throw "Cannot multiply matrices of incompatible dimensions.";
        }
        auto *product = new Array<T>(_numberOfRows, matrix.numberOfColumns());
        for (unsigned i = 0; i < _numberOfRows; i++) {
            for (unsigned j = 0; j < matrix.numberOfColumns(); j++) {
                for (unsigned k = 0; k < _numberOfColumns; k++) {
                    product->element(i, j) += element(i, k) * matrix.element(k, j);
                }
            }
        }
        return product;
    }

    // Adds an integer matrix into this matrix. Result stored in this matrix
    template<typename T>
    void Array<T>::AddIntoThis(const Array<T> &matrix) {
        if (_numberOfRows != matrix.numberOfRows() || _numberOfColumns != matrix.numberOfColumns() || _numberOfAisles != matrix.numberOfAisles()) {
            throw "Cannot add matrices of different dimensions.";
        }
        for (unsigned i = 0; i < size(); i++) {
            _array[i] += matrix._array[i];
        }
    }

    // Subtracts an integer matrix from this matrix. Result stored in this matrix
    template<typename T>
    void Array<T>::SubtractIntoThis(const Array<T> &matrix) {
        if (_numberOfRows != matrix.numberOfRows() || _numberOfColumns != matrix.numberOfColumns() || _numberOfAisles != matrix.numberOfAisles()) {
            throw "Cannot add matrices of different dimensions.";
        }
        for (unsigned i = 0; i < size(); i++) {
            _array[i] -= matrix._array[i];
        }
    }

    // Multiplies an integer matrix into this matrix. Result stored in this matrix
    template<typename T>
    void Array<T>::MultiplyIntoThis(const Array<T> &matrix) {
        throw("Not implemented yet.");
    }



    template<typename T>
    void Array<T>::HPCShitBoiiiii() {
        std::cout<<"First MPI shit boiiiiiiiiiii"<<std::endl;
        //std::cout<<"Number of available threads: "<< omp_get_max_threads ()<<std::endl;
    }

    template<typename T>
    void Array<T>::print() {
        for (unsigned i = 0; i < _numberOfRows; i++) {
            for (unsigned j = 0; j < _numberOfColumns; j++) {
                std::cout << _array[index(i, j)] << " ";
            }
            std::cout << std::endl;
        }
    }*/
