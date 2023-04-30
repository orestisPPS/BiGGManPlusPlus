
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


    template<typename T> class Array {

    public:
        //Custom constructor that takes the size of a 1D array 
        // and allocates the memory in the heap
        explicit Array(unsigned rows, bool isPositiveDefinite = false) :
                _numberOfRows(rows), _numberOfColumns(1), _numberOfAisles(1), parallelizationThreshold(1000), _isPositiveDefinite(isPositiveDefinite) {
            _array = new T[rows];
        }
        //Custom constructor that takes the size of a 2D array
        // and allocates the memory in the heap

        Array(unsigned rows, unsigned  columns, bool isPositiveDefinite = false) :
                _numberOfRows(rows), _numberOfColumns(columns), _numberOfAisles(1), parallelizationThreshold(1000), _isPositiveDefinite(isPositiveDefinite) {
            _array = new T[rows * columns];
        }
        //Custom constructor that takes the size of a 3D array
        // and allocates the memory in the heap

        Array(unsigned rows, unsigned  columns, unsigned aisles, bool isPositiveDefinite = false) :
                _numberOfRows(rows), _numberOfColumns(columns), _numberOfAisles(aisles), parallelizationThreshold(1000), _isPositiveDefinite(isPositiveDefinite) {
            _array = new T[rows * columns * aisles];
        }

        //Copy constructor
        Array(const Array<T>& matrix){
            _numberOfRows = matrix._numberOfRows;
            _numberOfColumns = matrix._numberOfColumns;
            _numberOfAisles = matrix._numberOfAisles;
            parallelizationThreshold = matrix.parallelizationThreshold;
            _isPositiveDefinite = matrix._isPositiveDefinite;

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
                throw out_of_range("matrix dimensions do not match");
            }
            _numberOfRows = matrix.numberOfRows();
            _numberOfColumns = matrix.numberOfColumns();
            _numberOfAisles = matrix.numberOfAisles();
            _isPositiveDefinite = matrix._isPositiveDefinite;
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
                auto *product = new Array<T>(_numberOfRows, matrix.numberOfColumns(), _numberOfAisles);
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

        // Swap the elements of the i-th and j-th rows of the matrix
        void swapRows(unsigned i, unsigned j) {
            if (i == j) return; // No need to swap if i and j are the same
            // Swap the elements of the i-th and j-th rows
            for (unsigned k = 0; k < _numberOfColumns; ++k) {
                T temp = (*this)(i, k);
                (*this)(i, k) = (*this)(j, k);
                (*this)(j, k) = temp;
            }
        }

        // Swap the elements of the i-th and j-th columns of the matrix
        void swapColumns(unsigned i, unsigned j) {
            if (i == j) return; // No need to swap if i and j are the same
            // Swap the elements of the i-th and j-th columns
            for (unsigned k = 0; k < _numberOfRows; ++k) {
                T temp = (*this)(k, i);
                (*this)(k, i) = (*this)(k, j);
                (*this)(k, j) = temp;
            }
        }

        T& vectorElement(unsigned i){
            if (i >= size()){
                throw out_of_range ("Index should be between 0 and " + to_string(size()));
            }
            return _array[i];
        }

        const T &vectorElement(unsigned i) const {
            if (i >= size()){
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

        bool isPositiveDefinite(){
            return isSymmetric() && isPositiveDefinite();
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

            for (unsigned i = 0; i < _numberOfRows; i++){
                for (unsigned j = 0; j < _numberOfColumns; j++){
                    transpose(j,i) = _array[i * _numberOfColumns + j];
                }
            }
            return transpose;
        }

        Array<T>* transposePtr(){
            auto transpose = new Array<T>(_numberOfColumns, _numberOfRows, _numberOfAisles);
            for (unsigned i = 0; i < _numberOfRows; i++){
                for (unsigned j = 0; j < _numberOfColumns; j++){
                    transpose->at(j,i) = _array[i * _numberOfColumns + j];
                }
            }
            return transpose;
        }

        // Stores the transpose of the matrix in the given matrix
        void transposeIntoThis(){
            auto *transpose = new Array<T>(_numberOfColumns, _numberOfRows);
            for (int i = 0; i < _numberOfRows; ++i) {
                for (int j = 0; j < _numberOfColumns; ++j) {
                    transpose->at(j, i, _array[i * _numberOfColumns + j]);
                }
            }
            _numberOfRows = transpose->numberOfRows();
            _numberOfColumns = transpose->numberOfColumns();
            _numberOfAisles = transpose->numberOfAisles();
            delete [] _array;
            _array = transpose->_array;
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
        vector<T> vectorMultiplication(vector<T> vector){
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


// SolverLUP decomposition of a square matrix using Doolittle's algorithm
// Returns a tuple with two Array<double> pointers for the decomposed matrices L and U
        tuple<Array<double>*, Array<double>*> LUdecomposition() {
            if (!isSquare()) {
                throw invalid_argument("The matrix is not square");
            }
            unsigned n = _numberOfRows;
            auto l = new Array<double>(n, n);
            auto u = new Array<double>(n, n);

            for (int i = 0; i < n; i++) {
                // Upper Triangular matrix U
                for (int j = i; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < i; k++) {
                        sum += l->at(i, k) * u->at(k, j);
                    }
                    u->at(i, j) = _array[i * n + j] - sum;
                }

                // Lower Triangular matrix L
                for (int j = i; j < n; j++) {
                    if (i == j) {
                        l->at(i, j) = 1;
                    } else {
                        double sum = 0;
                        for (int k = 0; k < i; k++) {
                            sum += l->at(j, k) * u->at(k, i);
                        }
                        l->at(j, i) = (_array[j * n + i] - sum) / u->at(i, i);
                    }
                }
            }

            auto LU = make_tuple(l, u);
            return LU;
        }

        //Stores the decomposed matrices L and U in this matrix
        void LUdecompositionOnMatrix(){
            if (!isSquare()){
                throw invalid_argument("The matrix is not square");
            }
            auto n = _numberOfRows;

            //March through rows of A and L
            for (int i = 0; i < n; ++i) {
                //Calculate Upper Triangular matrix U
                for (int j = i; j < n; j++) {
                    auto sum = 0.0;
                    //March through columns of L and rows of U
                    for (int k = 0; k < i; ++k) {
                        sum += _array[i * n + k] * _array[k * n + j];
                    }
                    _array[i * n + j] = _array[i * n + j] - sum;
                }
                //Calculate Lower Triangular matrix L
                for (int j = i + 1; j < n ; ++j) {
                    auto sum = 0.0;
                    for (int k = 0; k < i; ++k) {
                        sum += _array[j * n + k] * _array[k * n + i];
                    }
                    _array[j * n + i] = (_array[j * n + i] - sum) / _array[i * n + i];
                }
                _array[i * n + i] = 1;
            }
        }

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
        //Boolean that stores if the matrix is symmetric and has positive eigenvalues
        bool _isPositiveDefinite;
    };

} // Numerics

#endif //UNTITLED_ARRAY_H

