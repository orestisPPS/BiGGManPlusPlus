//
// Created by hal9000 on 11/26/22.
//
#include "Matrix.h"
#include <omp.h>

namespace Primitives {
    
    template <typename T>
    Array<T>::Array(size_t rows) {
        this->_numberOfRows = rows;
        this->_numberOfColumns = 1;
        this->_numberOfAisles = 1;
        _array = new T[rows];
    }

    template<class T>
    Array<T>::Array(size_t rows, size_t columns) {
        _numberOfRows = rows;
        _numberOfColumns = columns;
        _numberOfAisles = 1;
        _array = new T[_numberOfRows * _numberOfColumns];
    }
    template<class T>
    Array<T>::Array(size_t rows, size_t columns, size_t aisles) {
        _numberOfRows = rows;
        _numberOfColumns = columns;
        _numberOfAisles = aisles;
        _array = new T[_numberOfRows * _numberOfColumns * _numberOfAisles];
    }
    
    
    template<class T>
    Array<T>::Array(const Array<T> &matrix) {
        _numberOfRows = matrix._numberOfRows;
        _numberOfColumns = matrix._numberOfColumns;
        _array = new T[_numberOfRows * _numberOfColumns];
        for (int i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            _array[i] = matrix._array[i];
        }
    }
    
    template<class T>
    Array<T>::~Array() {
        delete[] _array;
        _array = nullptr;
    }

    template<class T>
    size_t Array<T>::numberOfRows() {
        return _numberOfRows;
    }

    template<class T>
    size_t Array<T>::numberOfColumns() {
        return _numberOfColumns;
    }
    
    template<class T>
    size_t Array<T>::numberOfAisles() {
        return _numberOfAisles;
    }
    
    template <typename T>
    size_t Array<T>::size() {
        return _numberOfRows * _numberOfColumns * _numberOfAisles;
    }
    
    template<class T>
    T Array<T>::element(size_t row) {
        return _array[row];
    }
    
    template<class T>
    T Array<T>::element(size_t row, size_t column) {
        return _array[row * _numberOfColumns + column];
    }
    
    template<class T>
    T Array<T>::element(size_t row, size_t column, size_t aisle) {
        return _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle];
    }
    
    template<class T>
    size_t Array<T>::index(size_t row) {
        return _array[row];
    }
    
    template<class T>
    size_t Array<T>::index(size_t row, size_t column) {
        return _array[row * _numberOfColumns + column];
    }
    
    template<class T>
    size_t Array<T>::index(size_t row, size_t column, size_t aisle) {
        return _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle];
    }

    template<class T>
    void Array<T>::populateElement(size_t row, T value) {
        _array[index(row)] = value;
    }

    template<class T>
    void Array<T>::populateElement(size_t row, size_t column, T value) {
        _array[index(row, column)] = value;
    }
    
    template<class T>
    void Array<T>::populateElement(size_t row, size_t column, size_t aisle, T value) {
        _array[index(row, column, aisle)] = value;
    }

    template<class T>
    bool Array<T>::isSquare() {
        return _numberOfRows == _numberOfColumns;
    }
    
    template<class T>
    bool Array<T>::isCubic() {
        return _numberOfRows == _numberOfColumns && _numberOfColumns == _numberOfAisles;
    }
    
    template <class T>
    bool isDiagonal() {
        return false;
    }
    
    template<class T>
    bool Array<T>::isVector() {
        return _numberOfColumns == _numberOfAisles == 1;
    }
    
    template<class T>
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
    
    template<class T>
    Array<T> Array<T>::transpose() {
        if (!isSquare() || isVector() == true || _numberOfAisles > 1) {
            throw "Cannot transpose a non-square matrix, a vector, or a matrix with more than one aisle.";
        }
        auto transpose = new Array<T>(_numberOfColumns, _numberOfRows);
        for (size_t i = 0; i < _numberOfRows; i++)
            for (size_t j = 0; j < _numberOfColumns; j++) {
                transpose->_array[transpose->index(j, i)] = _array[index(i, j)];
            }
        return transpose;
    }

    template<class T>
    void Array<T>::transposeIntoThis() {
        if (!isSquare() || isVector() == true || _numberOfAisles > 1) {
            throw "Cannot transpose a non-square matrix, a vector, or a matrix with more than one aisle.";
        }
        auto *transpose = new Array<T>(_numberOfColumns, _numberOfRows);
        for (size_t i = 0; i < _numberOfRows; i++) {
            for (size_t j = 0; j < _numberOfColumns; j++) {
                transpose->element(j, i) = element(index(i, j));
            }
        }
        _numberOfRows = transpose->_numberOfRows;
        _numberOfColumns = transpose->_numberOfColumns;
        delete[] _array;
        _array = transpose->_array;
    }
    
    // Overloaded assignment operator
    template<class T>
    Array<T>& Array<T>::operator=(const Array<T> &matrix) {
        if (this != &matrix) {
            this->~Array();
            this->_numberOfRows = matrix._numberOfRows;
            this->_numberOfColumns = matrix._numberOfColumns;
            this->_numberOfAisles = matrix._numberOfAisles;
            this->_array = new T[this-> _numberOfRows * this->_numberOfColumns];
            for (size_t i = 0; i <this-> _numberOfRows * this->_numberOfColumns; i++) {
                this->_array[i] = matrix._array[i];
            }
        }
        return *this;
    }

    // Overloaded equality operator
    template<class T>
    bool Array<T>::operator==(const Array<T> &matrix) {
        if (_numberOfRows != matrix.numberOfRows() || _numberOfColumns != matrix.numberOfColumns() || _numberOfAisles != matrix.numberOfAisles()) {
            return false;
        }
        for (size_t i = 0; i < size(); i++) {
            if (_array[i] != matrix._array[i]) {
                return false;
            }
        }
        return true;
    }

    // Overloaded inequality operator
    template<class T>
    bool Array<T>::operator!=(const Array<T> &matrix) {
        if (*this == matrix) {
            return false;
        }
    }
    //-----------------------------------------
    //----- Overloaded addition operator-------
    //-----------------------------------------
    
    // Overloaded operator for integer matrix addition
    template<> Array<int> Array<int>::operator+(const Array<int> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns || _numberOfAisles != matrix._numberOfAisles) {
            throw std::invalid_argument("Matrices are not of same size");
        }
        auto result = new Array<int>(_numberOfRows, _numberOfColumns, _numberOfAisles);
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++) {
            result->_array[i] = _array[i] + matrix._array[i];
        }
        return *result;
    }

    // Overloaded operator for double matrix addition
    template<> Array<double> Array<double>::operator+(const Array<double> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns|| _numberOfAisles != matrix._numberOfAisles) {
            throw std::invalid_argument("Array dimensions do not match");
        }
        auto *result = new Array<double>(_numberOfRows, _numberOfColumns, _numberOfAisles);
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++) {
            result->_array[i] = _array[i] + matrix._array[i];
        }
        return *result;
    }

    // Adds an integer matrix into this matrix. Result stored in this matrix
    template<> void Array<int>::AddMatrixIntoThis(const Array<int> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns || _numberOfAisles != matrix._numberOfAisles) {
            throw std::invalid_argument("Array dimensions do not match");
        }
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++) {
            _array[i] = _array[i] + matrix._array[i];
        }
    }

    // Adds a double matrix into this matrix. Result stored in this matrix
    template<> void Array<double>::AddMatrixIntoThis(const Array<double> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns || _numberOfAisles != matrix._numberOfAisles) {
            throw std::invalid_argument("Array dimensions do not match");
        }
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++) {
            _array[i] = _array[i] + matrix._array[i];
        }
    }

    //----------------------------------------
    //---- Overloaded subtraction operator----
    //----------------------------------------
    
    // Overloaded operator for integer matrix subtraction
    template<> Array<int> Array<int>::operator-(const Array<int> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns || _numberOfAisles != matrix._numberOfAisles) {
            throw std::invalid_argument("Matrices are not of same size");
        }
        auto result = new Array<int>(_numberOfRows, _numberOfColumns, _numberOfAisles);
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++) {
            result->_array[i] = _array[i] + matrix._array[i];
        }
        return *result;
    }

    // Overloaded operator for integer matrix subtraction
    template<> Array<double> Array<double>::operator-(const Array<double> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns|| _numberOfAisles != matrix._numberOfAisles) {
            throw std::invalid_argument("Array dimensions do not match");
        }
        auto *result = new Array<double>(_numberOfRows, _numberOfColumns, _numberOfAisles);
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++) {
            result->_array[i] = _array[i] + matrix._array[i];
        }
        return *result;
    }
    
    // Subtracts an integer matrix from this matrix. Result stored in this matrix
    template<> void Array<int>::SubtractMatrixIntoThis(const Array<int> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns || _numberOfAisles != matrix._numberOfAisles) {
            throw std::invalid_argument("Array dimensions do not match");
        }
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++) {
            _array[i] = _array[i] - matrix._array[i];
        }
    }
    
    // Subtracts a double matrix from this matrix. Result stored in this matrix
    template<> void Array<double>::SubtractMatrixIntoThis(const Array<double> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns || _numberOfAisles != matrix._numberOfAisles) {
            throw std::invalid_argument("Array dimensions do not match");
        }
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++) {
            _array[i] = _array[i] - matrix._array[i];
        }
    }
    
    template<class T>
    void Array<T>::HPCShitBoiiiii() {
        std::cout<<"First MPI shit boiiiiiiiiiii"<<std::endl;
        std::cout<<"Number of available threads: "<< omp_get_max_threads ()<<std::endl;
    }
    
    template<class T>
    void Array<T>::print() {
        for (size_t i = 0; i < _numberOfRows; i++) {
            for (size_t j = 0; j < _numberOfColumns; j++) {
                std::cout << _array[index(i, j)] << " ";
            }
            std::cout << std::endl;
        }
    }
}

/////OpenCL////////// GPU
////QT Framework 