//
// Created by hal9000 on 11/26/22.
//
#include "Matrix.h"
# include <omp.h>

namespace Primitives {

    template<class T>
    Matrix<T>::Matrix(size_t rows, size_t columns) {
        _numberOfRows = rows;
        _numberOfColumns = columns;
        _matrix = new T[_numberOfRows * _numberOfColumns];
    }

    template<class T>
    Matrix<T>::Matrix(const Matrix<T> &matrix) {
        _numberOfRows = matrix._numberOfRows;
        _numberOfColumns = matrix._numberOfColumns;
        _matrix = new T[_numberOfRows * _numberOfColumns];
        for (int i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            _matrix[i] = matrix._matrix[i];
        }
    }
    
    template<class T>
    Matrix<T>::~Matrix() {
        delete[] _matrix;
        _matrix = nullptr;
    }

    template<class T>
    size_t Matrix<T>::numberOfRows() {
        return _numberOfRows;
    }

    template<class T>
    size_t Matrix<T>::numberOfColumns() {
        return _numberOfColumns;
    }

    template<class T>
    size_t Matrix<T>::index(size_t row, size_t column) {
        return row * _numberOfColumns + column;
    }

    template<class T>
    T Matrix<T>::element(size_t row, size_t column) {
        return _matrix[index(row, column)];
    }

    template<class T>
    void Matrix<T>::populateElement(size_t row, size_t column, T value) {
        _matrix[index(row, column)] = value;
    }

    template<class T>
    bool Matrix<T>::isSquare() {
        return _numberOfRows == _numberOfColumns;
    }
    
    template<class T>
    bool Matrix<T>::isSymmetric() {
        if (!isSquare()) {
            return false;
        }
        for (int i = 0; i < _numberOfRows; i++) {
            for (int j = i; j < _numberOfColumns; j++) {
                if (_matrix[index(i, j)] != _matrix[index(j, i)]) {
                    return false;
                }
            }
        }
        return true;
    }
    
    template<class T>
    Matrix<T> Matrix<T>::transpose() {
        auto transpose = new Matrix<T>(_numberOfColumns, _numberOfRows);
        for (size_t i = 0; i < _numberOfRows; i++)
            for (size_t j = 0; j < _numberOfColumns; j++) {
                transpose->_matrix[transpose->index(j, i)] = _matrix[index(i, j)];
            }
        return transpose;
    }

    template<class T>
    void Matrix<T>::transposeIntoThis() {
        auto *transpose = new Matrix<T>(_numberOfColumns, _numberOfRows);
        for (size_t i = 0; i < _numberOfRows; i++) {
            for (size_t j = 0; j < _numberOfColumns; j++) {
                transpose->element(j, i) = element(index(i, j));
            }
        }
        _numberOfRows = transpose->_numberOfRows;
        _numberOfColumns = transpose->_numberOfColumns;
        delete[] _matrix;
        _matrix = transpose->_matrix;
    }
    
    // Overloaded assignment operator
    template<class T>
    Matrix<T>& Matrix<T>::operator=(const Matrix<T> &matrix) {
        if (this != &matrix) {
            this->~Matrix();
            this->_numberOfRows = matrix._numberOfRows;
            this->_numberOfColumns = matrix._numberOfColumns;
            this->_matrix = new T[this-> _numberOfRows * this->_numberOfColumns];
            for (size_t i = 0; i <this-> _numberOfRows * this->_numberOfColumns; i++) {
                this->_matrix[i] = matrix._matrix[i];
            }
        }
        return *this;
    }

    // Overloaded equality operator
    template<class T>
    bool Matrix<T>::operator==(const Matrix<T> &matrix) {
        if (_numberOfRows != matrix.numberOfRows() || _numberOfColumns != matrix.numberOfColumns()) {
            return false;
        }
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            if (_matrix[i] != matrix._matrix[i]) {
                return false;
            }
        }
        return true;
    }

    // Overloaded inequality operator
    template<class T>
    bool Matrix<T>::operator!=(const Matrix<T> &matrix) {
        if (*this == matrix) {
            return false;
        }
    }

    // Overloaded operator for integer matrix addition
    template<> Matrix<int> Matrix<int>::operator+(const Matrix<int> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns) {
            throw std::invalid_argument("Matrices are not of same size");
        }
        auto result = new Matrix<int>(_numberOfRows, _numberOfColumns);
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            result->_matrix[i] = _matrix[i] + matrix._matrix[i];
        }
        return *result;
    }

    // Overloaded operator for double matrix addition
    template<> Matrix<double> Matrix<double>::operator+(const Matrix<double> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        auto *result = new Matrix<double>(_numberOfRows, _numberOfColumns);
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            result->_matrix[i] = _matrix[i] + matrix._matrix[i];
        }
        return *result;
    }

    // Overloaded operator for integer matrix subtraction
    template<> Matrix<int> Matrix<int>::operator-(const Matrix<int> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        auto *result = new Matrix<int>(_numberOfRows, _numberOfColumns);
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            result->_matrix[i] = _matrix[i] - matrix._matrix[i];
        }
        return *result;
    }

    // Overloaded operator for double matrix subtraction 
    template<> Matrix<double> Matrix<double>::operator-(const Matrix<double> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        auto *result = new Matrix<double>(_numberOfRows, _numberOfColumns);
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            result->_matrix[i] = _matrix[i] - matrix._matrix[i];
        }
        return *result;
    }

    // Adds an integer matrix into this matrix. Result stored in this matrix
    template<> void Matrix<int>::AddMatrixIntoThis(const Matrix<int> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            _matrix[i] = _matrix[i] + matrix._matrix[i];
        }
    }

    // Adds an integer matrix into this matrix. Result stored in this matrix
    template<> void Matrix<double>::AddMatrixIntoThis(const Matrix<double> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            _matrix[i] = _matrix[i] + matrix._matrix[i];
        }
    }
    
    // Subtracts an integer matrix from this matrix. Result stored in this matrix
    template<> void Matrix<int>::SubtractMatrixIntoThis(const Matrix<int> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            _matrix[i] = _matrix[i] - matrix._matrix[i];
        }
    }
    
    // Subtracts a double matrix from this matrix. Result stored in this matrix
    template<> void Matrix<double>::SubtractMatrixIntoThis(const Matrix<double> &matrix) {
        if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        for (size_t i = 0; i < _numberOfRows * _numberOfColumns; i++) {
            _matrix[i] = _matrix[i] - matrix._matrix[i];
        }
    }
    
    template<class T>
    void Matrix<T>::HPCShitBoiiiii() {
        std::cout<<"First MPI shit boiiiiiiiiiii"<<std::endl;
        std::cout<<"Number of available threads: "<< omp_get_max_threads ()<<std::endl;
    }
    
    template<class T>
    void Matrix<T>::print() {
        for (size_t i = 0; i < _numberOfRows; i++) {
            for (size_t j = 0; j < _numberOfColumns; j++) {
                std::cout << _matrix[index(i, j)] << " ";
            }
            std::cout << std::endl;
        }
    }
}

/////OpenCL////////// GPU
////QT Framework 