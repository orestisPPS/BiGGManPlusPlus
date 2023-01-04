
//
// Created by hal9000 on 1/2/23.
//


#ifndef UNTITLED_ARRAY_H
#define UNTITLED_ARRAY_H

#include <iostream>
#include <vector>
using namespace std;

namespace LinearAlgebra {


    template<typename T> class Array {

    public:
        //Custom constructor that takes the size of a 1D array 
        // and allocates the memory in the heap
        Array(unsigned rows){
            this->_numberOfRows = rows;
            this->_numberOfColumns = 1;
            this->_numberOfAisles = 1;
            _array = new T[rows];
        }
        //Custom constructor that takes the size of a 2D array
        // and allocates the memory in the heap

        Array(unsigned rows, unsigned  columns){
            this->_numberOfRows = rows;
            this->_numberOfColumns = columns;
            this->_numberOfAisles = 1;
            _array = new T[rows * columns];
        }
        //Custom constructor that takes the size of a 3D array
        // and allocates the memory in the heap

        Array(unsigned rows, unsigned  columns, unsigned aisles){
            this->_numberOfRows = rows;
            this->_numberOfColumns = columns;
            this->_numberOfAisles = aisles;
            _array = new T[rows * columns * aisles];
        }

        //Copy constructor
        Array(const Array<T>& matrix){
            this->_numberOfRows = matrix._numberOfRows;
            this->_numberOfColumns = matrix._numberOfColumns;
            this->_numberOfAisles = matrix._numberOfAisles;
            _array = new T[matrix._numberOfRows * matrix._numberOfColumns * matrix._numberOfAisles];
            for (int i = 0; i < matrix._numberOfRows * matrix._numberOfColumns * matrix._numberOfAisles; ++i) {
                _array[i] = matrix._array[i];
            }
        }

        //Destructor
        ~Array(){
            delete[] _array;
            _array = nullptr;
        }

        // Overloaded equality operator
        bool operator == (const Array<T>& matrix){
            if (_numberOfRows == matrix.numberOfRows() && _numberOfColumns == matrix.numberOfColumns() && _numberOfAisles == matrix.numberOfAisles()){
                for (int i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; ++i) {
                    if (_array[i] != matrix.array()[i]){
                        return false;
                    }
                }
                return true;
            }
            return false;
        }

        // Overloaded inequality operator
        bool operator != (const Array<T>& matrix){
            if (_numberOfRows == matrix.numberOfRows() && _numberOfColumns == matrix.numberOfColumns() && _numberOfAisles == matrix.numberOfAisles()){
                for (int i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; ++i) {
                    if (_array[i] != matrix.array()[i]){
                        return true;
                    }
                }
                return false;
            }
            return true;
        }

        // Overloaded operator for matrix addition
        Array<T> operator + (const Array<T>& matrix){
            if (_numberOfRows == matrix.numberOfRows() && _numberOfColumns == matrix.numberOfColumns() && _numberOfAisles == matrix.numberOfAisles()){
                Array<T> *sum = new Array<T>(_numberOfRows, _numberOfColumns, _numberOfAisles);
                for (int i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; ++i) {
                    sum->populateElement(i, _array[i] + matrix.array()[i]);
                }
                return *sum;
            }
            return *this;
        }

        // Overloaded operator for matrix subtraction
        Array<T> operator - (const Array<T>& matrix){
            if (_numberOfRows == matrix.numberOfRows() && _numberOfColumns == matrix.numberOfColumns() && _numberOfAisles == matrix.numberOfAisles()){
                Array<T> *difference = new Array<T>(_numberOfRows, _numberOfColumns, _numberOfAisles);
                for (int i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; ++i) {
                    difference->populateElement(i, _array[i] - matrix.array()[i]);
                }
                return *difference;
            }
            return *this;
        }

        // Overloaded operator for matrix multiplication
        Array<T> operator * (const Array<T>& matrix){
            if (_numberOfColumns == matrix.numberOfRows()){
                Array<T> *product = new Array<T>(_numberOfRows, matrix.numberOfColumns(), _numberOfAisles);
                for (int i = 0; i < _numberOfRows; ++i) {
                    for (int j = 0; j < matrix.numberOfColumns(); ++j) {
                        T sum = 0;
                        for (int k = 0; k < _numberOfColumns; ++k) {
                            sum += _array[i * _numberOfColumns + k] * matrix.array()[k * matrix.numberOfColumns() + j];
                        }
                        product->populateElement(i, j, sum);
                    }
                }
                return *product;
            }
            return *this;
        }

        void AddIntoThis(const Array<T>& matrix){
            if (_numberOfRows == matrix.numberOfRows() && _numberOfColumns == matrix.numberOfColumns() && _numberOfAisles == matrix.numberOfAisles()){
                for (int i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; ++i) {
                    _array[i] += matrix.array()[i];
                }
            }
        }

        void SubtractIntoThis(const Array<T>& matrix){
            if (_numberOfRows == matrix.numberOfRows() && _numberOfColumns == matrix.numberOfColumns() && _numberOfAisles == matrix.numberOfAisles()){
                for (int i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; ++i) {
                    _array[i] -= matrix.array()[i];
                }
            }
        }

        void MultiplyIntoThis(const Array<T>& matrix){
            if (_numberOfColumns == matrix.numberOfRows()){
                Array<T> *product = new Array<T>(_numberOfRows, matrix.numberOfColumns(), _numberOfAisles);
                for (int i = 0; i < _numberOfRows; ++i) {
                    for (int j = 0; j < matrix.numberOfColumns(); ++j) {
                        T sum = 0;
                        for (int k = 0; k < _numberOfColumns; ++k) {
                            sum += _array[i * _numberOfColumns + k] * matrix.array()[k * matrix.numberOfColumns() + j];
                        }
                        product->populateElement(i, j, sum);
                    }
                }
                _numberOfRows = product->numberOfRows();
                _numberOfColumns = product->numberOfColumns();
                _numberOfAisles = product->numberOfAisles();
                delete [] _array;
                _array = product->array();
            }
        }

        //Number of Rows. Array size : Height
        unsigned numberOfRows(){
            return _numberOfRows;
        }

        //Number of Columns.Array size : Width
        unsigned numberOfColumns(){
            return _numberOfColumns;
        }

        //Number of Aisles. Array size : Depth
        unsigned numberOfAisles(){
            return _numberOfAisles;
        }

        //Returns the size or the array
        unsigned size(){
            return _numberOfRows * _numberOfColumns * _numberOfAisles;
        }

        // Returns the value at the given row
        T& element(unsigned row){
            return _array[row];
        }

        // Returns the value at the given row and column
        T& element(unsigned row, unsigned column){
            return _array[row * _numberOfColumns + column];
        }

        // Returns the value at the given row, column and aisle
        T& element(unsigned row, unsigned column, unsigned aisle){
            return _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle];
        }

        //Populates the array element by value
        void populateElement(unsigned row, T value){
            _array[row] = value;
        }

/*        //Populates the array element by reference
        void populateElement(unsigned row, T &value){
            _array[row] = value;
        }

        //Populates the array element by pointer
        void populateElement(unsigned row, T *value){
            _array[row] = *value;
        }*/

        //Populates the array element by value
        void populateElement(unsigned row, unsigned column, T value){
            _array[row * _numberOfColumns + column] = value;
        }

/*        //Populates the array element by reference
        void populateElement(unsigned row, unsigned column, T &value){
            _array[row * _numberOfColumns + column] = value;
        }

        //Populates the array element by pointer
        void populateElement(unsigned row, unsigned column, T *value){
            _array[row * _numberOfColumns + column] = *value;
        }*/

        //Populates the array element by value
        void populateElement(unsigned row, unsigned column, unsigned aisle, T value){
            _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle] = value;
        }

/*        //Populates the array element by reference
        void populateElement(unsigned row, unsigned column, unsigned aisle, T &value){
            _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle] = value;
        }

        //Populates the array element by pointer
        void populateElement(unsigned row, unsigned column, unsigned aisle, T *value){
            _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle] = *value;
        }*/

        //Returns the array element by value
        T element_byValue (unsigned row){
            return _array[row];
        }

        //Returns array[row] by reference
        T &element_byReference (unsigned row){
            return &_array[row];
        }

        //Returns array[row] by pointer
        T *element_byPointer (unsigned row){
            return *_array[row];
        }

        //Returns the array element by value
        T element_byValue (unsigned row, unsigned column){
            return _array[row * _numberOfColumns + column];
        }

        //Returns array[row][column] by reference
        T &element_byReference (unsigned row, unsigned column){
            return &_array[row * _numberOfColumns + column];
        }

        //Returns array[row][column] by pointer
        T *element_byPointer (unsigned row, unsigned column){
            return *_array[row * _numberOfColumns + column];
        }

        //Returns the array element by value
        T element_byValue (unsigned row, unsigned column, unsigned aisle){
            return _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle];
        }

        //Returns array[row][column][aisle] by reference
        T &element_byReference (unsigned row, unsigned column, unsigned aisle){
            return &_array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle];
        }

        //Returns array[row][column][aisle] by pointer
        T *element_byPointer (unsigned row, unsigned column, unsigned aisle){
            return *_array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle];
        }


        //Boolean defining if 2d array is square
        bool isSquare(){
            return _numberOfRows == _numberOfColumns;
        }

        //Boolean defining if the 3d array is cubic
        bool isCubic(){
            return _numberOfRows == _numberOfColumns && _numberOfColumns == _numberOfAisles;
        }

        //Boolean defining if the matrix is a vector
        bool isVector(){
            return _numberOfColumns == _numberOfAisles == 1;
        }

        //Boolean defining if the matrix is symmetric
        bool isSymmetric(){
            if (isSquare()){
                for (int i = 0; i < _numberOfRows; ++i) {
                    for (int j = i; j < _numberOfColumns; ++j) {
                        if (_array[i * _numberOfColumns + j] != _array[j * _numberOfColumns + i]){
                            return false;
                        }
                    }
                }
                return true;
            }
            return false;
        }

        //Boolean defining if the matrix is diagonal
        bool isDiagonal(){
            if (isSquare()){
                for (int i = 0; i < _numberOfRows; ++i) {
                    for (int j = 0; j < _numberOfColumns; ++j) {
                        if (i != j && _array[i * _numberOfColumns + j] != 0){
                            return false;
                        }
                    }
                }
                return true;
            }
            return false;
        }

        // Returns the pointer of the transpose of the matrix
        Array<T> *transpose(){
            auto *transpose = new Array<T>(_numberOfColumns, _numberOfRows);
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    transpose->populateElement(j, i, _array[i * _numberOfColumns + j]);
                }
            }
            return transpose;
        }

        // Stores the transpose of the matrix in the given matrix
        void transposeIntoThis(){
            auto *transpose = new Array<T>(_numberOfColumns, _numberOfRows);
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    transpose->populateElement(j, i, _array[i * _numberOfColumns + j]);
                }
            }
            _numberOfRows = transpose->numberOfRows();
            _numberOfColumns = transpose->numberOfColumns();
            _numberOfAisles = transpose->numberOfAisles();
            delete [] _array;
            _array = transpose->array();
        }

        // Overloaded assignment operator
        Array<T>& operator = (const Array<T>& matrix){
            if (this != &matrix){
                _numberOfRows = matrix.numberOfRows();
                _numberOfColumns = matrix.numberOfColumns();
                _numberOfAisles = matrix.numberOfAisles();
                delete [] _array;
                _array = new T[_numberOfRows * _numberOfColumns * _numberOfAisles];
                for (int i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; ++i) {
                    _array[i] = matrix.array()[i];
                }
            }
            return *this;
        }

        vector<T> getRow(unsigned row){
            vector<T> rowVector;
            for (int i = 0; i < _numberOfColumns; ++i) {
                rowVector.push_back(_array[row * _numberOfColumns + i]);
            }
            return rowVector;
        }

        vector<T> getColumn(unsigned column){
            vector<T> columnVector;
            for (int i = 0; i < _numberOfRows; ++i) {
                columnVector.push_back(_array[i * _numberOfColumns + column]);
            }
            return columnVector;
        }

        vector<T> getAisle(unsigned aisle){
            vector<T> aisleVector;
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    aisleVector.push_back(_array[i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + aisle]);
                }
            }
            return aisleVector;
        }



        void HPCShitBoiiiii(){
            throw "not implemented";
        }

        // Prints the matrix in the console
        void print(){
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    std::cout << _array[i * _numberOfColumns + j] << " ";
                }
                std::cout << std::endl;
            }
        }

    private:
        // The 1D array that stores the matrix. It is stored in the heap
        T* _array;
        //Number of Rows. Array size : Height
        unsigned _numberOfRows;
        //Number of Columns.Array size : Width
        unsigned _numberOfColumns;
        //Number of Aisles. Array size : Depth
        unsigned _numberOfAisles;
    };

} // Numerics

#endif //UNTITLED_ARRAY_H
