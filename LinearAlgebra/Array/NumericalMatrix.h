//
// Created by hal9000 on 8/16/23.
//

#ifndef UNTITLED_NUMERICALMATRIX_H
#define UNTITLED_NUMERICALMATRIX_H

#include <vector>
#include <stdexcept>
#include <memory>
#include <type_traits>
#include "../ParallelizationMethods.h"
using namespace std;

namespace LinearAlgebra {

    /**
     * @brief Represents a numerical matrix.
     * 
     * This class is designed to handle matrices with various data types including raw pointers, smart pointers, and objects.
     * 
     * @tparam T Type of data used to store matrix elements.
     */
    template <typename T>
    class NumericalMatrix {

    public:

        /**
         * @brief Constructs a new NumericalMatrix object.
         * 
         * @param rows Number of rows for the matrix.
         * @param columns Number of columns for the matrix.
         * @param initialValue Default value for matrix elements.
         * @param parallelizationMethod Parallelization method to be used for matrix operations.
         */
        explicit NumericalMatrix(unsigned int rows, unsigned int columns, T initialValue = 0, ParallelizationMethod parallelizationMethod = SingleThread) :
                _numberOfRows(rows), _numberOfColumns(columns), _parallelizationMethod(parallelizationMethod) {
            _values = new vector<T>(_numberOfRows * _numberOfColumns, initialValue);
        }

        /**
         * @brief Deletes the NumericalMatrix object and deallocates the data vector pointer
         */
        ~NumericalMatrix(){
            _values->clear();
            delete _values;
            _values = nullptr;
        }

        /**
         * @brief Copy constructor for NumericalMatrix.
         * 
         * This copy constructor can handle raw pointers, smart pointers, and other heap objects.
         * 
         * @param other The matrix to be copied from.
         */
        NumericalMatrix(const NumericalMatrix& other) :
                _numberOfRows(other._numberOfRows), _numberOfColumns(other._numberOfColumns), _parallelizationMethod(other._parallelizationMethod) {
            _values = copyData(other._values);
        }

        /**
        * @brief Element access operator for the matrix.
        * 
        * @param row Row index.
        * @param col Column index.
        * @return T& Reference to the element at the specified position.
        */
        T& operator()(unsigned int row, unsigned int col) {
            return (*_values)[row * _numberOfColumns + col];
        }

        /**
     * @brief Overloads the assignment operator for NumericalMatrix.
     * 
     * @param other The matrix to be assigned from.
     * @return NumericalMatrix& Reference to the current matrix after assignment.
     */
        NumericalMatrix& operator=(const NumericalMatrix& other) {
            if (this != &other) {  // protect against self-assignment
                delete _values;
                _numberOfRows = other._numberOfRows;
                _numberOfColumns = other._numberOfColumns;
                _parallelizationMethod = other._parallelizationMethod;
                _values = copyData(other._values);
            }
            return *this;
        }

        /**
         * @brief Checks if two matrices are equal.
         * 
         * @param other The matrix to be compared with.
         * @return bool True if matrices are equal, false otherwise.
         */
        bool operator==(const NumericalMatrix& other) const {
            if (_numberOfRows != other._numberOfRows || _numberOfColumns != other._numberOfColumns) {
                return false;
            }

            auto itOther = other._values->begin();
            for (auto & val : *_values) {
                if (val != *itOther) {
                    return false;
                }
                ++itOther;
            }
            return true;
        }


        /**
         * @brief Checks if two matrices are not equal.
         * 
         * @param other The matrix to be compared with.
         * @return bool True if matrices are not equal, false otherwise.
         */
        bool operator!=(const NumericalMatrix& other) const {
            return !(*this == other);
        }


        /**
         * @brief Gets the data pointer of the matrix.
         * 
         * @return T* Pointer to the matrix data.
         */
        T* getDataPointer() const {
            return _values->data();
        }

    private:

        vector<T> * _values; ///< Pointer to the data storing the matrix elements.
        
        unsigned int _numberOfRows; ///< Number of rows in the matrix.
        
        unsigned int _numberOfColumns; ///< Number of columns in the matrix.
        
        ParallelizationMethod _parallelizationMethod; ///< Parallelization method used for matrix operations.
        
        /**
         * @brief Deep copies data for raw pointers.
         * 
         * @param source The source data to be copied from.
         * @return U Deep copied data.
         */
        template <typename U = T>
        typename enable_if<is_pointer<U>::value, U>::type copyData(const U& source) {
            U newData = new vector<typename remove_pointer<U>::type>(*source);
            return newData;
        }

        /**
         * @brief Copies data for std::shared_ptr.
         * 
         * @param source The source data to be copied from.
         * @return U Copied shared_ptr data.
         */
        template <typename U = T>
        typename enable_if<is_same<U, shared_ptr<vector<typename U::element_type>>>::value, U>::type copyData(const U& source) {
            return source;
        }

        /**
         * @brief Copies data for other objects using the copy constructor.
         * 
         * @param source The source data to be copied from.
         * @return U Copied data.
         */
        template <typename U = T>
        typename enable_if<!is_pointer<U>::value && !is_same<U, shared_ptr<vector<typename U::element_type>>>::value, U>::type copyData(const U& source) {
            return U(source);
        }


    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALMATRIX_H
