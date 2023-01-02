//
// Created by hal9000 on 11/25/22.
//
#pragma once
#include <iostream>
#include <map>
#include <omp.h>

namespace Primitives {
    template<class T>
    class Array {
        
        public:
            //Custom constructor that takes the size of a 1D array 
            // and allocates the memory in the heap
            Array(size_t rows);
            //Custom constructor that takes the size of a 2D array
            // and allocates the memory in the heap

            Array(size_t rows, size_t  columns);
            //Custom constructor that takes the size of a 3D array
            // and allocates the memory in the heap

            Array(size_t rows, size_t  columns, size_t aisles);
            
            //Copy constructor

            Array(const Array<T>& matrix);
            
            //Destructor
            ~Array();
            
            //Number of Rows. Array size : Height
            size_t numberOfRows();
            
            //Number of Columns.Array size : Width
            size_t numberOfColumns();
            
            //Number of Aisles. Array size : Depth
            size_t numberOfAisles();
            
            //Returns the size or the array
            size_t size();
            
            // Returns the value at the given row
            T element(size_t row);
            
            // Returns the value at the given row and column
            T element(size_t row, size_t column);
            
            // Returns the value at the given row, column and aisle
            T element(size_t row, size_t column, size_t aisle);
            
            //Sets the value at the given row
            void populateElement(int row, T value);
            
            //Sets the value at the given row and column
            void populateElement(size_t row, size_t column, T value);
            
            //Sets the value at the given row, column and aisle
            void populateElement(size_t row, size_t column, size_t aisle, T value);
            
            //Boolean defining if 2d array is square
            bool isSquare();
            
            //Boolean defining if the 3d array is cubic
            bool isCubic();
            
            //Boolean defining if the matrix is a vector
            bool isVector();
            
            //Boolean defining if the matrix is symmetric
            bool isSymmetric();
            
            //Boolean defining if the matrix is diagonal
            bool isDiagonal();
            
            // Returns the pointer of the transpose of the matrix
            Array<T> transpose();
            
            // Stores the transpose of the matrix in the given matrix
            void transposeIntoThis();
            
            // Overloaded assignment operator
            Array<T>& operator = (const Array<T>& matrix);
            
            // Overloaded equality operator
            bool operator == (const Array<T>& matrix);
            
            // Overloaded inequality operator
            bool operator != (const Array<T>& matrix);
            
            // Overloaded operator for matrix addition
            Array<T> operator + (const Array<T>& matrix);

            // Overloaded operator for matrix subtraction
            Array<T> operator - (const Array<T>& matrix);
    
            // Overloaded operator for matrix multiplication
            Array<T> operator * (const Array<T>& matrix);
                        
            void AddIntoThis(const Array<T>& matrix);
            
            void SubtractIntoThis(const Array<T>& matrix);
            
            void MultiplyIntoThis(const Array<T>& matrix);
            
            void HPCShitBoiiiii();
            
            // Prints the matrix in the console
            void print();
    
        private:
            // The 1D array that stores the matrix. It is stored in the heap
            T* _array;
            //Number of Rows. Array size : Height
            size_t _numberOfRows;
            //Number of Columns.Array size : Width
            size_t _numberOfColumns;
            //Number of Aisles. Array size : Depth
            size_t _numberOfAisles;
            // Finds the index of the 1D array in the private memory efficient 1D array
            size_t index(size_t row);            
            // Finds the index of the 2D array in the private memory efficient 1D array
            size_t index(size_t row, size_t column);
            // Finds the index of the 3D array in the private memory efficient 1D array 
            size_t index(size_t row, size_t column, size_t aisle);
        };
    }
    // LinearAlgebra
