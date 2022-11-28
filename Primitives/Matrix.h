//
// Created by hal9000 on 11/25/22.
//


#pragma once
#include <iostream>
namespace Primitives {

    template<class T>
    class Matrix {
        public:
            //Custom constructor that takes the size of the matrix 
            // and allocates the memory in the heap
            Matrix(size_t rows, size_t  columns);
            
            //Copy constructor
            Matrix(const Matrix<T>& matrix);
            
            //Destructor
            ~Matrix();
            
            //Number of Rows. Matrix size : Height
            size_t numberOfRows();
            
            //Number of Columns.Matrix size : Width
            size_t numberOfColumns();
            
            // Returns the value at the given row and column
            T element(size_t row, size_t column);
            
            //Sets the value at the given row and column
            void populateElement(size_t row, size_t column, T value);
            
            //Boolean defining if the matrix is square
            bool isSquare();
            
            //Boolean defining if the matrix is symmetric
            bool isSymmetric();
            
            // Returns the pointer of the transpose of the matrix
            Matrix<T> transpose();
            
            // Stores the transpose of the matrix in the given matrix
            void transposeIntoThis();
            
            // Overloaded assignment operator
            Matrix<T>& operator = (const Matrix<T>& matrix);
            
            // Overloaded equality operator
            bool operator == (const Matrix<T>& matrix);
            
            // Overloaded inequality operator
            bool operator != (const Matrix<T>& matrix);
            
            // Overloaded operator for integer matrix addition
            Matrix<int> operator + (const Matrix<int>& matrix);
                       
            // Overloaded operator for double addition
            Matrix<double> operator + (const Matrix<double>& matrix);
            
        // Overloaded operator for integer matrix subtraction
            Matrix<int> operator - (const Matrix<int>& matrix);
            
            // Overloaded operator for double matrix subtraction
            Matrix<double> operator - (const Matrix<double>& matrix);
            
            // Overloaded operator for integer matrix multiplication
            Matrix<int> operator * (const Matrix<int>& matrix);
            
            // Overloaded operator for double matrix multiplication
            Matrix<double> operator * (const Matrix<double>& matrix);
            
            void AddMatrixIntoThis(const Matrix<int>& matrix);
            
            void AddMatrixIntoThis(const Matrix<double>& matrix);

            void SubtractMatrixIntoThis(const Matrix<int>& matrix);
            
            void SubtractMatrixIntoThis(const Matrix<double>& matrix);
            
            void HPCShitBoiiiii();
            
            // Prints the matrix in the console
            void print();
    
        private:
            // The 1D array that stores the matrix. It is stored in the heap
            T* _matrix;
            //Number of Rows. Matrix size : Height
            size_t _numberOfRows;
            //Number of Columns.Matrix size : Width
            size_t _numberOfColumns;
            // Finds the index of the 2D array in the private memory efficient 1D array
            size_t index(size_t row, size_t column);
        };
    } // LinearAlgebra
