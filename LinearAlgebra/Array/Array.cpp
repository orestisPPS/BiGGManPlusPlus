//
// Created by hal9000 on 5/4/23.
//
#include "Array.h"

namespace LinearAlgebra {

    template<typename T>
    Array<T>::Array(short unsigned numberOfRows, short unsigned numberOfColumns, short unsigned numberOfAisles,
                    T initialValue, bool isPositiveDefinite) :
            _numberOfRows(numberOfRows), _numberOfColumns(numberOfColumns), _numberOfAisles(numberOfAisles),
            _array(vector<T>(numberOfRows * numberOfColumns * numberOfAisles, initialValue)),
            _isPositiveDefinite(isPositiveDefinite), _isSquare(false),
            parallelizationThreshold(1E4) {
        if (numberOfRows == numberOfColumns and numberOfColumns == numberOfAisles)
            _isSquare = true;
    }

    template<typename T>
    Array<T>::Array(const Array<T> &array) :
            _numberOfRows(array._numberOfRows), _numberOfColumns(array._numberOfColumns),
            _numberOfAisles(array._numberOfAisles),
            _array(array._array), _isPositiveDefinite(array._isPositiveDefinite), _isSquare(array._isSquare),
            parallelizationThreshold(array.parallelizationThreshold) {}

    template<typename T>
    T &Array<T>::operator()(unsigned i) {
        if (i >= _numberOfRows * _numberOfColumns * _numberOfAisles)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfAisles == 1 && _numberOfColumns == 1)
            return _array[i];
        else
            throw invalid_argument("The matrix is not one-dimensional.");
    }

    template<typename T>
    const T &Array<T>::operator()(unsigned i) const {
        if (i >= _numberOfRows * _numberOfColumns * _numberOfAisles)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfAisles == 1 and _numberOfColumns == 1)
            return _array[i];
        else
            throw invalid_argument("The matrix is not one-dimensional.");
    }

    template<typename T>
    T &Array<T>::operator()(unsigned i, unsigned j) {
        if (i >= _numberOfRows or j >= _numberOfColumns)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfAisles == 1)
            return _array[i * _numberOfColumns + j];
        else
            throw invalid_argument("The matrix is not two-dimensional.");
    }

    template<typename T>
    const T &Array<T>::operator()(unsigned i, unsigned j) const {
        if (i >= _numberOfRows or j >= _numberOfColumns)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles == 1)
            return _array[i * _numberOfColumns + j];
        else
            throw invalid_argument("The matrix is not two-dimensional.");
    }

    template<typename T>
    T &Array<T>::operator()(unsigned i, unsigned j, unsigned k) {
        if (i >= _numberOfRows or j >= _numberOfColumns)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles == 1)
            return _array[i * _numberOfColumns + j];
        else
            throw invalid_argument("The matrix is not two-dimensional.");
    }

    template<typename T>
    const T &Array<T>::operator()(unsigned i, unsigned j, unsigned k) const {
        if (i >= _numberOfRows or j >= _numberOfColumns or k >= _numberOfAisles)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles > 1)
            return _array[i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k];
        else
            throw invalid_argument("The matrix is not three-dimensional.");
    }

    template<typename T>
    T &Array<T>::at(unsigned i) {
        if (i >= _numberOfRows * _numberOfColumns * _numberOfAisles)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfAisles == 1 and _numberOfColumns == 1)
            return _array[i];
        else
            throw invalid_argument("The matrix is not one-dimensional.");
    }

    template<typename T>
    const T &Array<T>::at(unsigned i) const {
        if (i >= _numberOfRows * _numberOfColumns * _numberOfAisles)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfAisles == 1 and _numberOfColumns == 1)
            return _array[i];
        else
            throw invalid_argument("The matrix is not one-dimensional.");
    }

    template<typename T>
    T &Array<T>::at(unsigned i, unsigned j) {
        if (i >= _numberOfRows or j >= _numberOfColumns)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles == 1)
            return _array[i * _numberOfColumns + j];
        else
            throw invalid_argument("The matrix is not two-dimensional.");
    }

    template<typename T>
    const T &Array<T>::at(unsigned i, unsigned j) const {
        if (i >= _numberOfRows or j >= _numberOfColumns)
            throw invalid_argument("The index is out of bounds.");
        if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles == 1)
            return _array[i * _numberOfColumns + j];
        else
            throw invalid_argument("The matrix is not two-dimensional."); 
    }

    template<typename T>
    T &Array<T>::at(unsigned i, unsigned j, unsigned k) {
        if (i >= _numberOfRows or j >= _numberOfColumns or k >= _numberOfAisles)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles > 1)
            return _array[i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k];
        else
            throw invalid_argument("The matrix is not three-dimensional.");
    }

    template<typename T>
    const T &Array<T>::at(unsigned i, unsigned j, unsigned k) const {
        if (i >= _numberOfRows or j >= _numberOfColumns or k >= _numberOfAisles)
            throw out_of_range("The index is out of bounds.");
        if (_numberOfRows > 1 and _numberOfColumns > 1 and _numberOfAisles > 1)
            return _array[i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + k];
        else
            throw invalid_argument("The matrix is not three-dimensional.");
    }

    template<typename T>
    Array<T> &Array<T>::operator=(const Array<T> &array) {
        if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
            _numberOfAisles != array._numberOfAisles)
            throw invalid_argument("The dimensions of the arrays are not the same.");
        _isSquare = array._isSquare;
        _isPositiveDefinite = array._isPositiveDefinite;
        for (int i = 0; i < array._array.size(); ++i) {
            _array[i] = array._array[i];
        }
        return *this;
    }
    
    template<typename T>
    bool Array<T>::operator==(const Array<T> &array) const {
        if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
            _numberOfAisles != array._numberOfAisles)
            return false;
        for (int i = 0; i < array._array.size(); ++i) {
            if (_array[i] != array._array[i])
                return false;
        }
        return true;
    }
    
    template<typename T>
    bool Array<T>::operator!=(const Array<T> &array) const {
        return !(*this == array);
    }
    
    template<typename T>
    Array<T> Array<T>::add(const Array<T>& array) const{
        if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
            _numberOfAisles != array._numberOfAisles)
            throw invalid_argument("The dimensions of the arrays are not the same.");
        Array<T> result(_numberOfRows, _numberOfColumns, _numberOfAisles);
        for (int i = 0; i < _array.size(); ++i) {
            result._array[i] = _array[i] + array._array[i];
        }
        return result;
    }
    
    template<typename T>
    void Array<T>::addIntoThis(const Array<T>& array){
        if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
            _numberOfAisles != array._numberOfAisles)
            throw invalid_argument("The dimensions of the arrays are not the same.");
        for (int i = 0; i < _array.size(); ++i) {
            _array[i] += array._array[i];
        }
    }
    
    template<typename T>
    Array<T> Array<T>::subtract(const Array<T>& array) const{
        if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
            _numberOfAisles != array._numberOfAisles)
            throw invalid_argument("The dimensions of the arrays are not the same.");
        Array<T> result(_numberOfRows, _numberOfColumns, _numberOfAisles);
        for (int i = 0; i < _array.size(); ++i) {
            result._array[i] = _array[i] - array._array[i];
        }
        return result;
    }
    
    template<typename T>
    void Array<T>::subtractIntoThis(const Array<T>& array){
        if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
            _numberOfAisles != array._numberOfAisles)
            throw invalid_argument("The dimensions of the arrays are not the same.");
        for (int i = 0; i < _array.size(); ++i) {
            _array[i] -= array._array[i];
        }
    }
    
    template<typename T>
    Array<T> Array<T>::multiply(const Array<T>& array, unsigned minRow, unsigned maxRow, unsigned minCol, unsigned maxCol) const{
        if (_numberOfRows != array._numberOfRows or _numberOfColumns != array._numberOfColumns or
            _numberOfAisles != array._numberOfAisles)
            throw invalid_argument("The dimensions of the arrays are not the same.");
        if (minRow > maxRow or minCol > maxCol)
            throw invalid_argument("The range is invalid.");
        
    }
    
    template<typename T>
    Array<T> Array<T>::multiply(const Array<T>& array) const{
        return multiply(array, 0, _numberOfRows - 1, 0, _numberOfColumns - 1);
    }
    
    template<typename T>
    vector<T> Array<T>::multiplyWithVector(const vector<T> &vector) const {
        if (_numberOfColumns != vector.size())
            throw invalid_argument("The dimensions of the array and the vector are not the same.");
        auto result = std::vector<T>(_numberOfRows);
        for (int i = 0; i < _numberOfRows; ++i) {
            for (int j = 0; j < _numberOfColumns; ++j) {
                result[i] += _array[i * _numberOfColumns + j] * vector[j];
            }
        }
    }
    
    template<typename T>
    Array<T> Array<T>::transpose() const {
        if (_numberOfRows != _numberOfColumns)
            throw invalid_argument("The matrix is not square.");
        for (int i = 0; i < _numberOfRows; ++i) {
            for (int j = i + 1; j < _numberOfColumns; ++j) {
                swap(_array[i * _numberOfColumns + j], _array[j * _numberOfColumns + i]);
            }
        }
    }
    
    template<typename T>
    void Array<T> ::transposeIntoThis(){
        if (_numberOfRows != _numberOfColumns)
            throw invalid_argument("The matrix is not square.");
        for (int i = 0; i < _numberOfRows; ++i) {
            for (int j = i + 1; j < _numberOfColumns; ++j) {
                swap(_array[i * _numberOfColumns + j], _array[j * _numberOfColumns + i]);
            }
        }
    }
    
    template<typename T>
    bool Array<T>::isSquare() const {
        return _isSquare;
    }
    
    template<typename T>
    bool Array<T>::isSymmetric() const {
        if (_numberOfRows != _numberOfColumns)
            throw invalid_argument("The matrix is not square.");
        for (int i = 0; i < _numberOfRows; ++i) {
            for (int j = i + 1; j < _numberOfColumns; ++j) {
                if (_array[i * _numberOfColumns + j] != _array[j * _numberOfColumns + i])
                    return false;
            }
        }
        return true;
    }
    
    template<typename T>
    bool Array<T>::isPositiveDefinite() const {
        return _isPositiveDefinite;
    }
    
    template<typename T>
    void Array<T>::setPositiveDefinite(bool isPositiveDefinite) {
        _isPositiveDefinite = isPositiveDefinite;
    }
    
    template<typename T>
    bool Array<T>::isDiagonal() const {
        for (int i = 0; i < size(); ++i) {
            for (int j = 0; j < size(); ++j) {
                if (i != j and _array[i * _numberOfColumns + j] != 0)
                    return false;
            }
        }
    }

    //Number of Rows. Array size : Height
    template<typename T>
    unsigned Array<T>::numberOfRows() const {
        return _numberOfRows;
    }

    //Number of Columns.Array size : Width
    template<typename T>
    unsigned Array<T>::numberOfColumns() const {
        return _numberOfColumns;
    }

    //Number of Aisles. Array size : Depth
    template<typename T>
    unsigned Array<T>::numberOfAisles() const {
        return _numberOfAisles;
    }
    //Returns the size or the array
    template<typename T>
    unsigned int Array<T>::size() {
        return _array.size();
    }
    
    template<typename T>
    void Array<T>::swapRows(unsigned int i, unsigned int j) {
        if (i == j) return; // No need to swap if i and j are the same
        // Swap the elements of the i-th and j-th rows
        for (auto k = 0; k < _numberOfColumns; k++) {
            T temp = (*this)(i, k);
            (*this)(i, k) = (*this)(j, k);
            (*this)(j, k) = temp;
        }
    }
    
    template<typename T>
    void Array<T>::swapColumns(unsigned int i, unsigned int j) {
        if (i == j) return; // No need to swap if i and j are the same
        // Swap the elements of the i-th and j-th columns
        for (auto k = 0; k < _numberOfRows; k++) {
            T temp = (*this)(k, i);
            (*this)(k, i) = (*this)(k, j);
            (*this)(k, j) = temp;
        }
    }

    template<typename T>
    vector<T> Array<T>::getRow(unsigned row){
        vector<T> rowVector;
        for (int i = 0; i < _numberOfColumns; ++i) {
            rowVector.push_back(_array[row * _numberOfColumns + i]);
        }
        return rowVector;
    }

    template<typename T>
    vector<T> Array<T>::getColumn(unsigned column){
        vector<T> columnVector;
        for (int i = 0; i < _numberOfRows; ++i) {
            columnVector.push_back(_array[i * _numberOfColumns + column]);
        }
        return columnVector;
    }

    template<typename T>
    vector<T> Array<T>::getAisle(unsigned aisle){
        vector<T> aisleVector;
        for (int i = 0; i < _numberOfRows; ++i) {
            for (int j = 0; j < _numberOfColumns; ++j) {
                aisleVector.push_back(_array[i * _numberOfColumns * _numberOfAisles + j * _numberOfAisles + aisle]);
            }
        }
        return aisleVector;
    }
    
    template<typename T>
    void Array<T>::print() const {
        for (int i = 0; i < _numberOfRows; ++i) {
            for (int j = 0; j < _numberOfColumns; ++j) {
                cout << _array[i * _numberOfColumns + j] << " ";
            }
            cout << endl;
        }
    }
        

}