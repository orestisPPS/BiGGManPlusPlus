
//
// Created by hal9000 on 1/2/23.
//


#ifndef UNTITLED_ARRAY_H
#define UNTITLED_ARRAY_H

#include <iostream>
#include <vector>
#include <limits>
#include <omp.h>

using namespace std;

namespace LinearAlgebra {


    template<typename T> class Array {

    public:
        //Custom constructor that takes the size of a 1D array 
        // and allocates the memory in the heap
        Array(unsigned rows) : _numberOfRows(rows), _numberOfColumns(1), _numberOfAisles(1), parallelizationThreshold(1000) {
            _array = new T[rows];
        }
        //Custom constructor that takes the size of a 2D array
        // and allocates the memory in the heap

        Array(unsigned rows, unsigned  columns) :
        _numberOfRows(rows), _numberOfColumns(columns), _numberOfAisles(1), parallelizationThreshold(1000) {
            _array = new T[rows * columns];
        }
        //Custom constructor that takes the size of a 3D array
        // and allocates the memory in the heap

        Array(unsigned rows, unsigned  columns, unsigned aisles) :
        _numberOfRows(rows), _numberOfColumns(columns), _numberOfAisles(aisles), parallelizationThreshold(1000){
            _array = new T[rows * columns * aisles];
        }

        //Copy constructor
        Array(const Array<T>& matrix){
            _numberOfRows = matrix._numberOfRows;
            _numberOfColumns = matrix._numberOfColumns;
            _numberOfAisles = matrix._numberOfAisles;
            parallelizationThreshold = matrix.parallelizationThreshold;
            
            _array = new T[matrix._numberOfRows * matrix._numberOfColumns * matrix._numberOfAisles];
            if (size() >= parallelizationThreshold){
/*                #pragma omp parallel for default(none) 
                for (unsigned i = 0; i < _numberOfRows * _numberOfColumns; i++){
                    _array[i] = matrix._array[i];
                }*/
            }
            else{
                for (unsigned i = 0; i < size(); i++){
                    _array[i] = matrix._array[i];
                }
            }
        }
        
        //Destructor
        ~Array(){
            delete[] _array;
            _array = nullptr;
        }
        
        unsigned parallelizationThreshold;
        
        // () operator overloading.
        // Allows to set the elements of the matrix and returns a reference to the element at position (i)
        T& operator()(unsigned i) {
            return _array[i];
        }
        
        // () operator overloading.
        // Allows to read the elements of the matrix and returns a constant reference to the element at position (i)
        const T& operator()(unsigned i) const {
            return _array[i];
        }
        
        // () operator overloading.
        // Allows to set the elements of the matrix and returns a reference to the element at position (i, j)
        T& operator()(unsigned i, unsigned j) {
            return _array[i * _numberOfColumns + j];
        }
        
        // () operator overloading.
        // Allows to read the elements of the matrix and returns a constant reference to the element at position (i, j)
        const T& operator()(unsigned i, unsigned j) const {
            return _array[i * _numberOfColumns + j];
        }
        
        // () operator overloading.
        // Allows to set the elements of the matrix and returns a reference to the element at position (i, j, k)
        T& operator()(unsigned i, unsigned j, unsigned k) {
            return _array[i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k];
        }
        
        // () operator overloading.
        // Allows to read the elements of the matrix and returns a constant reference to the element at position (i, j, k)
        const T& operator()(unsigned i, unsigned j, unsigned k) const {
            return _array[i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k];
        }
        
        //->at operator
        // Allows to set the elements of the matrix and returns a reference to the element at position (i)
        T& at(unsigned i) {
            if (i >= size()){
                throw out_of_range("Index out of range");
            }
            return _array[i];
        }
        
        T& at(unsigned i) const {
            if (i >= size()){
                throw out_of_range("Index out of range");
            }
            return _array[i];
        }
        

        T& at(unsigned i, unsigned j) {
            if (i >= size()){
                throw out_of_range("Index out of range");
            }
            return _array[ i * _numberOfColumns + j];
        }
        
        T& at(unsigned i, unsigned j) const {
            if (i >= size()){
                throw out_of_range("Index out of range");
            }
            return _array[ i * _numberOfColumns + j];
        }

        T& at(unsigned i, unsigned j, unsigned k) {
            if (i >= size()){
                throw out_of_range("Index out of range");
            }
            return _array[ i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k];
        }
        
        T& at(unsigned i, unsigned j, unsigned k) const {
            if (i >= size()){
                throw out_of_range("Index out of range");
            }
            return _array[ i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k];
        }
        
        
        
        

        // Overloaded assignment operator
        Array<T>& operator = (const Array<T>& matrix){
            if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns || _numberOfAisles != matrix._numberOfAisles) {
                throw out_of_range("Matrix dimensions do not match");
            }
                _numberOfRows = matrix.numberOfRows();
                _numberOfColumns = matrix.numberOfColumns();
                _numberOfAisles = matrix.numberOfAisles();
                delete [] _array;
                _array = new T[_numberOfRows * _numberOfColumns * _numberOfAisles];
                auto size = this->size();
                if (this->size() >= parallelizationThreshold){
/*                    #pragma omp parallel for  firstprivate(matrix) default(none) 
                    for (unsigned i = 0; i < size(); i++){
                        _array[i] = matrix.vectorElement(i);
                    }*/
                }
                else{
                    for (unsigned i = 0; i < _numberOfRows * _numberOfColumns; i++){
                        _array[i] = matrix._array[i];
                    }
                }
            return *this;
        }
        
        // Overloaded equality operator
        bool operator == (const Array<T>& matrix) const {
            if (_numberOfRows != matrix._numberOfRows || _numberOfColumns != matrix._numberOfColumns || _numberOfAisles != matrix._numberOfAisles){
                return false;
            }
            if (this->size() >= parallelizationThreshold){
/*                auto areEqual = true;
                #pragma omp parallel for default(none) 
                for (unsigned i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++){
                    if (_array[i] != matrix._array[i]){
                        areEqual = false;
                    }
                }
                return areEqual;*/
            }
            else{
                for (unsigned i = 0; i < _numberOfRows * _numberOfColumns; i++){
                    if (_array[i] != matrix._array[i]){
                        return false;
                    }
                }
            }
        }

        // Overloaded inequality operator
        bool operator != (const Array<T>& matrix) const {
            return !(*this == matrix);
        }

        // Overloaded operator for matrix addition
        Array<T> operator + (const Array<T>& matrix) const {
            if (_numberOfRows != matrix._numberOfRows && _numberOfColumns != matrix._numberOfColumns && _numberOfAisles != matrix._numberOfAisles){
               throw out_of_range("The dimensions of the matrices do not match");
            }
            Array<T> result(_numberOfRows, _numberOfColumns, _numberOfAisles);

            if (_numberOfRows * _numberOfColumns >= parallelizationThreshold){
/*                #pragma omp parallel for default(none) firstprivate(result)
                for (unsigned i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++){
                    result._array[i] = _array[i] + matrix.vectorElement(i);
                }*/
            }
            else{
                for (unsigned i = 0; i < _numberOfRows * _numberOfColumns; i++){
                    result._array[i] = _array[i] + matrix._array[i];
                }
            }
            return result;
        }
        
        // Overloaded operator for matrix subtraction
        Array<T> operator - (const Array<T>& matrix) const {
            if (_numberOfRows != matrix.numberOfRows() && _numberOfColumns != matrix.numberOfColumns() && _numberOfAisles != matrix.numberOfAisles()){
                throw out_of_range("The dimensions of the matrices do not match");
            }
            Array<T> result(_numberOfRows, _numberOfColumns, _numberOfAisles);
            if (this->size() >= parallelizationThreshold){
/*                #pragma omp parallel for default(none) firstprivate(result)
                for (unsigned i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++){
                    result._array[i] = _array[i] - matrix.vectorElement(i);
                }*/
            }
            else{
                for (unsigned i = 0; i < _numberOfRows * _numberOfColumns; i++){
                    result._array[i] = _array[i] - matrix.vectorElement(i);
                }
            }
            return result;
        }
        
        // Overloaded operator for matrix multiplication
        Array<T> operator * (const Array<T>& matrix) const {
            if (_numberOfColumns != matrix.numberOfRows()){
                throw out_of_range("The dimensions of the matrices do not match");
            }
            Array<T> result(_numberOfRows, matrix.numberOfColumns(), _numberOfAisles);
            if (this->size() >= parallelizationThreshold){
/*                #pragma omp parallel for default(none)
                for (unsigned i = 0; i < _numberOfRows; i++){
                    for (unsigned j = 0; j < matrix.numberOfColumns(); j++){
                        for (unsigned k = 0; k < _numberOfColumns; k++){
                            result._array[i * matrix.numberOfColumns() + j] += _array[i * _numberOfColumns + k] * matrix._array[k * matrix.numberOfColumns() + j];
                        }
                    }
                }*/
            }
            else{
                for (unsigned i = 0; i < _numberOfRows; i++){
                    for (unsigned j = 0; j < matrix.numberOfColumns(); j++){
                        for (unsigned k = 0; k < _numberOfColumns; k++){
                            result._array[i * matrix.numberOfColumns() + j] += _array[i * _numberOfColumns + k] * matrix._array[k * matrix.numberOfColumns() + j];
                        }
                    }
                }
            }
            return result;
        }
        
        // Overloaded operator for matrix multiplication with a scalar
        Array<T> operator * (const T& scalar) const {
            Array<T> result(_numberOfRows, _numberOfColumns, _numberOfAisles);
            if (this->size() >= parallelizationThreshold){
/*                #pragma omp parallel for default(none)
                for (unsigned i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++){
                    result._array[i] = _array[i] * scalar;
                }*/
            }
            else{
                for (unsigned i = 0; i < _numberOfRows * _numberOfColumns; i++){
                    result._array[i] = _array[i] * scalar;
                }
            }
            return result;
        }
        
        // Overloaded operator for matrix division by a scalar
        Array<T> operator / (const T& scalar) const {
            Array<T> result(_numberOfRows, _numberOfColumns, _numberOfAisles);
            if (size() >= parallelizationThreshold){/*
                #pragma omp parallel for default(none)
                for (unsigned i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; i++){
                    result._array[i] = _array[i] / scalar;
                }*/
            }
            else{
                for (unsigned i = 0; i < size(); i++){
                    result._array[i] = _array[i] / scalar;
                }
            }
            return result;
        }
        
        void AddIntoThis(const Array<T>& matrix){
            if (_numberOfRows == matrix.numberOfRows() && _numberOfColumns == matrix.numberOfColumns() && _numberOfAisles == matrix.numberOfAisles()){
                for (int i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; ++i) {
                    _array[i] += matrix.vectorElement(i);
                }
            }
        }

        void SubtractIntoThis(const Array<T>& matrix){
            if (_numberOfRows == matrix.numberOfRows() && _numberOfColumns == matrix.numberOfColumns() && _numberOfAisles == matrix.numberOfAisles()){
                for (int i = 0; i < _numberOfRows * _numberOfColumns * _numberOfAisles; ++i) {
                    _array[i] -= matrix.vectorElement(i);
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

        T& vectorElement(unsigned i){
            if (i >= size() || i < 0){
                throw out_of_range ("Index should be between 0 and " + to_string(size()));
            }
            return _array[i];
        }
        
        const T &vectorElement(unsigned i) const {
            if (i >= size() || i < 0){
                throw out_of_range ("Index should be between 0 and " + to_string(size()));
            }
            return _array[i];
        }

        
        //Number of Rows. Array size : Height
        const unsigned &numberOfRows() const {
            return _numberOfRows;
        }

        //Number of Columns.Array size : Width
        const unsigned &numberOfColumns() const {
            return _numberOfColumns;
        }

        //Number of Aisles. Array size : Depth
        const unsigned &numberOfAisles() const {
            return _numberOfAisles;
        }
        //Returns the size or the array
        unsigned int size() {
            return _numberOfRows * _numberOfColumns * _numberOfAisles;
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
        Array<T> transpose(){
            Array<T> transpose(_numberOfColumns, _numberOfRows, _numberOfAisles);
            if (size() >= parallelizationThreshold){
                #pragma omp parallel for default(none)
                for (unsigned i = 0; i < _numberOfRows; i++){
                    for (unsigned j = 0; j < _numberOfColumns; j++){
                        transpose(j,i) = _array[i * _numberOfColumns + j];
                    }
                }
            }
            else{
                for (unsigned i = 0; i < _numberOfRows; i++){
                    for (unsigned j = 0; j < _numberOfColumns; j++){
                        transpose(j,i) = _array[i * _numberOfColumns + j];
                    }
                }
            }
            return *transpose;
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
        
#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
        //If the vector size is bigger than 1000 the multiplication will be performed in parallel with OpenMP
        vector<T> vectorMultiplication(vector<T> vector, bool diagonal = false
            if (vector.size() == _numberOfColumns){
                ::vector<T> resultVector;
                if (vector.size() > 1000){
                    #pragma omp parallel for
                    for (int i = 0; i < _numberOfRows; ++i) {
                        T result = 0;
                        for (int j = 0; j < _numberOfColumns; ++j) {
                            result += _array[i * _numberOfColumns + j] * vector[j];
                        }
                        resultVector.push_back(result);
                    }    // NOLINT(openmp-use-default-none)
                } else {
                    for (int i = 0; i < _numberOfRows; ++i) {
                        T result = 0;
                        for (int j = 0; j < _numberOfColumns; ++j) {
                            //add boolean about diagonal here
                            result += _array[i * _numberOfColumns + j] * vector[j];
                        }
                        resultVector.push_back(result);
                    }
                }
                return resultVector;
            }
            return vector;
        }
#pragma clang diagnostic pop
            


        // Prints the matrix in the console
        void print(){
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    cout << _array[i * _numberOfColumns + j] << " ";
                }
                cout << endl;
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



/*
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

*/
/*        //Populates the array element by reference
        void populateElement(unsigned row, T &value){
            _array[row] = value;
        }

        //Populates the array element by pointer
        void populateElement(unsigned row, T *value){
            _array[row] = *value;
        }*//*


        //Populates the array element by value
        void populateElement(unsigned row, unsigned column, T value){
            _array[row * _numberOfColumns + column] = value;
        }

*/
/*        //Populates the array element by reference
        void populateElement(unsigned row, unsigned column, T &value){
            _array[row * _numberOfColumns + column] = value;
        }

        //Populates the array element by pointer
        void populateElement(unsigned row, unsigned column, T *value){
            _array[row * _numberOfColumns + column] = *value;
        }*//*


        //Populates the array element by value
        void populateElement(unsigned row, unsigned column, unsigned aisle, T value){
            _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle] = value;
        }

*/
/*        //Populates the array element by reference
        void populateElement(unsigned row, unsigned column, unsigned aisle, T &value){
            _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle] = value;
        }

        //Populates the array element by pointer
        void populateElement(unsigned row, unsigned column, unsigned aisle, T *value){
            _array[row * _numberOfColumns * _numberOfAisles + column * _numberOfAisles + aisle] = *value;
        }*//*


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
*/
