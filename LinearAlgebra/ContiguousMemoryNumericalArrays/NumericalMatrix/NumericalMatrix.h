//
// Created by hal9000 on 8/16/23.
//

#ifndef UNTITLED_NUMERICALMATRIX_H
#define UNTITLED_NUMERICALMATRIX_H

#include "../NumericalVector/NumericalVector.h"
#include "MatrixStorageDataProviders/CSRStorageDataProvider.h"
#include "MatrixStorageDataProviders/FullMatrixStorageDataProvider.h"
#include "NumericalMatrixMathematicalOperations/FullMatrixMathematicalOperationsProvider.h"
#include "NumericalMatrixMathematicalOperations/CSRMathematicalOperationsProvider.h"
#include "NumericalMatrixMathematicalOperations/EigendecompositionProvider.h"
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
        explicit NumericalMatrix(unsigned int rows, unsigned int columns, NumericalMatrixStorageType storageType = FullMatrix,
                                 unsigned availableThreads = 1) :
                _numberOfRows(rows), _numberOfColumns(columns), _availableThreads(availableThreads){
            dataStorage = _initializeStorage(storageType);
            _math = _initializeMath();
        }

        //TODO FIX THIS
        /**
        * Copy constructor for NumericalMatrix.
        *          * This copy constructor can handle raw pointers, smart pointers, and other heap objects.
        * 
        * @param other The matrix to be copied from.
        */
        template<typename InputType>
        explicit NumericalMatrix(const InputType& other) {
            _checkInputMatrixDataType(other);
            _numberOfRows = dereference_trait<InputType>::numberOfRows(other);
            _numberOfColumns = dereference_trait<InputType>::numberOfColumns(other);
            _availableThreads = dereference_trait<InputType>::getAvailableThreads(other);
            
            dataStorage = _initializeStorage(dereference_trait<InputType>::getStorageType(other));
            auto otherStorage = other.storage;

            auto thisValues = dataStorage->getValues();
            auto otherValues = otherStorage->getValues();
            (*thisValues) = (*otherValues);
            
            auto thisSupplementaryVectors = dataStorage->getSupplementaryVectors();
            auto otherSupplementaryVectors = otherStorage->getSupplementaryVectors();
            if (otherSupplementaryVectors.size() > 0){
                for (unsigned i = 0; i < otherSupplementaryVectors.size(); ++i){
                    thisSupplementaryVectors[i] = otherSupplementaryVectors[i];
                }
            }

            _math = _initializeMath();
        }
        

        /**
         * @brief Move constructor for NumericalMatrix.
         * @param other 
         */
        NumericalMatrix(NumericalMatrix&& other) noexcept :
                _numberOfRows(std::move(other._numberOfRows)),
                _numberOfColumns(std::move(other._numberOfColumns)),
                _availableThreads(std::move(other._availableThreads)),
                dataStorage(std::move(other.dataStorage)),
                _math(_initializeMath())
        {
        }

        
        shared_ptr<NumericalMatrixStorageDataProvider<T>> dataStorage; ///< Storage object for the matrix elements.


        /**
        * @brief Move assignment operator for NumericalMatrix.
        * 
        * Efficiently transfers the resources of the given matrix to the current matrix.
        * 
        * @param other The matrix to be moved.
        * @return NumericalMatrix& Reference to the current matrix after the move.
        */
        NumericalMatrix& operator=(NumericalMatrix&& other) noexcept {
            if (this != &other) {
                //_values = std::move(other._values);
                _numberOfRows = std::exchange(other._numberOfRows, 0);
                _numberOfColumns = std::exchange(other._numberOfColumns, 0);
                dataStorage = std::move(other.dataStorage);
                _availableThreads = std::exchange(other._availableThreads, 0);
            }
            return *this;
        }

        /**
         * @brief Checks if two matrices are equal.
         * 
         * @param other The matrix to be compared with.
         * @return bool True if matrices are equal, false otherwise.
         */
        template<typename InputType>
        bool operator==(const InputType& other) const {
            _checkInputMatrixDataType(other);
            if (numberOfRows() != dereference_trait<InputType>::numberOfRows(other) ||
                numberOfColumns() != dereference_trait<InputType>::numberOfColumns(other)) {
                return false;
            }
            const T *otherStorage = dereference_trait<InputType>::getDataStorageNumericalVectors(other);
            
            dataStorage->areElementsEqual(otherStorage);
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

        //=================================================================================================================//
        //=================================================== Data Access =================================================//
        //=================================================================================================================//
        
        T& getElement(unsigned row, unsigned column){
            return dataStorage->getElement(row, column);
        }
        
        void setElement(unsigned row, unsigned column, const T &value){
            dataStorage->setElement(row, column, value);
        }
        
        void eraseElement(unsigned row, unsigned column, const T &value){
            dataStorage->eraseElement(row, column, value);
        }

        /**
         * @brief Gets the number of rows in the matrix.
         * 
         * @return unsigned int Number of rows.
         */
        unsigned int numberOfRows() const {
            return _numberOfRows;
        }
        
        /**
         * @brief Gets the number of columns in the matrix.
         * 
         * @return unsigned int Number of columns.
         */
        unsigned int numberOfColumns() const {
            return _numberOfColumns;
        }
        
        /**
         * @brief Gets the size of the matrix.
         * 
         * @return unsigned int Size of the matrix.
         */
        unsigned int size() const {
            return _numberOfRows * _numberOfColumns;
        }


        /**
        * @brief Checks if this vector is empty.
        * @return true if the vector is empty, false otherwise.
        */
        bool isEmpty() const {
            return dataStorage->storage->_values->empty();
        }

        /**
        * @brief Fills this matrix with the specified value.
        * @param value The value to fill the matrix with.
        */
        void fill(T value) {
            dataStorage->_values->fill(value);
        }


        /**
        * \brief Fills this matrix with random values between the specified minimum and maximum.
        * 
        * This function uses the Mersenne Twister random number generator to generate 
        * random values. The generator is seeded with a device-dependent random number 
        * to ensure different sequences in different program runs.
        * 
        * \tparam T The data type of the vector components (e.g., double, float).
        * 
        * \param min The minimum value for the random numbers.
        * \param max The maximum value for the random numbers.
        */
        void fillRandom(T min, T max) {
            dataStorage->_values->fillRandom(min, max);
        }

        /**
        * @brief Gets the data pointer of this matrix.
        * 
        * @return T* Pointer to the matrix data.
        */


        /**
        * @brief Calculates the sum of all the elements of this matrix.
        * @return The sum of all components of the vector.
        * 
        * Given a vector v = [v1, v2, ..., vn] that contains all the matrix elements in row major format, the sum is calculated as:
        * sum(v) = v1 + v2 + ... + vn
        */
        T sum() const {
            return dataStorage->storage->_values->sum();
        }

        /**
        * @brief Calculates the sum of all the elements of this matrix.
        * @return The sum of all components of the vector.
        * 
        * Given a vector v = [v1, v2, ..., vn] that contains all the matrix elements in row major format, the sum is calculated as:
        * sum(v) = v1 + v2 + ... + vn
        */
        double average() {
            return dataStorage->_values->average();
        }

        //=================================================================================================================//
        //================================================== Matrix Operations ============================================//
        //=================================================================================================================//
        
        /**
        * @brief Scales all elements of this matrix.
        * Given matrix A = [a1, a2, ..., an] and scalar factor a, the updated matrix is:
        * A = [a*a1, a*a2, ..., a*an].
        * 
        * @param scalar The scaling factor.
        */
        void scale(T scalar){
            dataStorage->_values->scale(scalar);
        }

        /**
        * \brief Performs element-wise addition of two scaled matrices.
        * Given two vectors A = [v1, v2, ..., vn] and B = [w1, w2, ..., wn] representing the matrices in row major format,
         * and scalar factors a and b, their addition is:
        * add(A, B) = [a*A11+b*B11, a*A12+b*w12, ..., a*Anm+b*Bnm].
        * 
        * \param inputMatrix The input matrix to add.
        * \param result The result matrix after addition.
        * \param scaleThis Scaling factor for the current vector (default is 1).
        * \param scaleInput Scaling factor for the input vector (default is 1).
        */
       template<typename InputMatrixType1, typename InputMatrixType2>
        void add(const InputMatrixType1 &inputMatrix, InputMatrixType2 &resultMatrix, T scaleThis = 1, T scaleInput = 1) {

            _checkInputMatrixDataType(inputMatrix);
            _checkInputMatrixDimensions(inputMatrix);
            _checkInputMatrixStorageType(inputMatrix);
            _checkInputMatrixDataType(resultMatrix);
            _checkInputMatrixDimensions(resultMatrix);
            _checkInputMatrixStorageType(resultMatrix);
            
            auto inputStorage = dereference_trait<InputMatrixType1>::getDataStorageNumericalVectors(inputMatrix);
            auto resultStorage = dereference_trait<InputMatrixType2>::getDataStorageNumericalVectors(resultMatrix);

            _math->matrixAddition(inputStorage, resultStorage, scaleThis, scaleInput);
        }
        
        /**
        * \brief Performs element-wise subtraction of two scaled matrices.
        * Given two vectors A = [A11, A12, ..., Anm] and B = [B11, B12, ..., Bnm] representing the matrices in row major format,
        * and scalar factors a and b, their subtraction is:
        * add(A, B) = [a*A11-b*B11, a*A12-b*B12, ..., a*Anm-b*Bnm].
        * 
        * \param inputMatrix The input matrix to subtract.
        * \param resultMatrix The resultMatrix matrix after subtraction.
        * \param scaleThis Scaling factor for the current vector (default is 1).
        * \param scaleInput Scaling factor for the input vector (default is 1).
        */
        template<typename InputMatrixType1, typename InputMatrixType2>
        void subtract(const InputMatrixType1 &inputMatrix, InputMatrixType2 &resultMatrix, T scaleThis = 1, T scaleInput = 1) {

            _checkInputMatrixDataType(inputMatrix);
            _checkInputMatrixDimensions(inputMatrix);
            _checkInputMatrixStorageType(inputMatrix);
            _checkInputMatrixDataType(resultMatrix);
            _checkInputMatrixDimensions(resultMatrix);
            _checkInputMatrixStorageType(resultMatrix);
            
            auto inputStorage = dereference_trait<InputMatrixType1>::getDataStorageNumericalVectors(inputMatrix);
            auto resultStorage = dereference_trait<InputMatrixType2>::getDataStorageNumericalVectors(resultMatrix);
            
            _math->matrixSubtraction(inputStorage, resultStorage, scaleThis, scaleInput);
        }
        
        /**
         * @brief Performs matrix multiplication of two matrices.
         * 
         * @param inputMatrix The input matrix to multiply.
         * @param resultMatrix The result matrix after multiplication.
         * @param scaleThis Scaling factor for the current vector (default is 1).
         * @param scaleInput Scaling factor for the input vector (default is 1).
         */
        template<typename InputMatrixType1, typename InputMatrixType2>
        void multiplyMatrix(const InputMatrixType1 &inputMatrix, InputMatrixType2 &resultMatrix, T scaleThis = 1, T scaleInput = 1) {
            
            _checkInputMatrixDataType(inputMatrix);
            _checkInputMatrixStorageType(inputMatrix);
            _checkInputMatrixDataType(resultMatrix);
            _checkInputMatrixStorageType(resultMatrix);
            if (_numberOfColumns != dereference_trait<InputMatrixType1>::numberOfColumns(inputMatrix))
                throw invalid_argument("Input matrix must have the same number of columns as the current matrix.");
            if (_numberOfRows != dereference_trait<InputMatrixType1>::numberOfRows(inputMatrix))
                throw invalid_argument("Input matrix must have the same number of rows as the current matrix.");
            
            auto inputStorage = dereference_trait<InputMatrixType1>::getDataStorageNumericalVectors(inputMatrix);
            auto resultStorage = dereference_trait<InputMatrixType2>::getDataStorageNumericalVectors(resultMatrix);

            _math->matrixMultiplication(inputStorage, resultStorage, scaleThis, scaleInput);
        }
        
        /**
         * @brief Performs matrix-vector multiplication.
         * 
         * @param inputVector The input vector to multiply.
         * @param resultVector The result vector after multiplication.
         * @param scaleThis Scaling factor for the current vector (default is 1).
         * @param scaleInput Scaling factor for the input vector (default is 1).
         */
        template<typename InputVectorType1, typename InputVectorType2>
        void multiplyVector(const InputVectorType1 &inputVector, const InputVectorType2 &resultVector, T scaleThis = 1, T scaleInput = 1) {
            
            _checkInputVectorDataType(inputVector);
            _checkInputVectorDataType(resultVector);
            if (_numberOfColumns != dereference_trait_vector<InputVectorType1>::size(inputVector))
                throw invalid_argument("Input vector must have the same number of columns as the current matrix.");
            if (dereference_trait_vector<InputVectorType1>::size(inputVector) != dereference_trait_vector<InputVectorType2>::size(resultVector))
                throw invalid_argument("Input vector must have the same number of rows as the result vector.");
            auto inputVectorData = dereference_trait_vector<InputVectorType1>::dereference(inputVector);
            auto resultVectorData = dereference_trait_vector<InputVectorType2>::dereference(resultVector);
            
            _math->matrixVectorMultiplication(inputVectorData, resultVectorData, scaleThis, scaleInput);
        }
         

        
    private:
        
        unsigned int _numberOfRows; ///< Number of rows in the matrix.
        
        unsigned int _numberOfColumns; ///< Number of columns in the matrix.
        
        unsigned _availableThreads; ///< Number of available threads for parallelization.
        
        unique_ptr<NumericalMatrixMathematicalOperationsProvider<T>> _math; ///< Mathematical operations provider for the matrix.

        template<typename InputMatrixType>
        bool _checkInputMatrixDimensions(const InputMatrixType &inputMatrix) const {
            if (numberOfRows() != dereference_trait<InputMatrixType>::numberOfRows(inputMatrix) ||
                numberOfColumns() != dereference_trait<InputMatrixType>::numberOfColumns(inputMatrix)) {
                return false;
            }
            return true;
        }

        template<typename InputMatrixType>
        void _checkInputMatrixDataType(const InputMatrixType &input) {
            static_assert(std::is_same<InputMatrixType, NumericalMatrix<T>>::value
                          || std::is_same<InputMatrixType, std::shared_ptr<NumericalMatrix<T>>>::value
                          || std::is_same<InputMatrixType, std::unique_ptr<NumericalMatrix<T>>>::value
                          || std::is_same<InputMatrixType, NumericalMatrix<T>*>::value,
                          "Input must be a NumericalMatrix, its pointer, or its smart pointers.");
        }
        
        template<typename InputMatrixType>
        void _checkInputMatrixStorageType(const InputMatrixType &input) {
            auto inputStorage = dereference_trait<InputMatrixType>::getDataStorageNumericalVectors(input);
            if (dataStorage->getStorageType() != dereference_trait<InputMatrixType>::getStorageType(input))
                throw std::invalid_argument("Input matrix must be stored in the same format as the current matrix.");
        }
        
        template<typename InputVectorType>
        void _checkInputVectorDataType(const InputVectorType &input) {
            static_assert(std::is_same<InputVectorType, NumericalVector<T>>::value
                          || std::is_same<InputVectorType, std::shared_ptr<NumericalVector<T>>>::value
                          || std::is_same<InputVectorType, std::unique_ptr<NumericalVector<T>>>::value
                          || std::is_same<InputVectorType, NumericalVector<T>*>::value,
                          "Input must be a NumericalVector, its pointer, or its smart pointers.");
        }
        
        shared_ptr<NumericalMatrixStorageDataProvider<T>> _initializeStorage(NumericalMatrixStorageType storageType){
            switch (storageType) {
                case FullMatrix:
                    return make_shared<FullMatrixStorageDataProvider<T>>(_numberOfRows, _numberOfColumns, _availableThreads);
                case CSR:
                    return make_shared<CSRStorageDataProvider<T>>(_numberOfRows, _numberOfColumns, _availableThreads);
                default:
                    throw std::invalid_argument("Invalid storage type.");
            }
        }
        
        unique_ptr<FullMatrixMathematicalOperationsProvider<T>> _initializeMath(){
            switch (dataStorage->getStorageType()) {
                case FullMatrix:
                    return make_unique<FullMatrixMathematicalOperationsProvider<T>>(_numberOfRows, _numberOfColumns, dataStorage);
                    break;
                case CSR:
                    return make_unique<FullMatrixMathematicalOperationsProvider<T>>(_numberOfRows, _numberOfColumns, dataStorage);
                    break;
                default:
                    throw std::invalid_argument("Invalid storage type.");
                
            }
        }
        
        
        
        //=================================================================================================================//
        //============================================ Dereference Traits =================================================//
        //=================================================================================================================//

        /**
        * \brief Trait to standardize dereferencing of various types.
        *
        * This trait provides a unified way to dereference types such as raw pointers,
        * unique pointers, shared pointers, and direct objects.
        */
        template<typename U>
        struct dereference_trait;

        /**
        * \brief Base trait for raw pointers and direct objects.
        *
        * This trait provides a unified way to dereference types like NumericalVector and 
        * raw pointers to NumericalVector.
        */
        template<typename U>
        struct dereference_trait_base {
            /**
            * \brief Dereferences the source.
            * \param source A pointer to the source object.
            * \return A pointer to the data of the source.
            */
            static U *dereference(U *source) {
                static_assert(std::is_arithmetic<U>::value, "Template type T must be an arithmetic type (integral or floating-point)");

                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->getDataPointer();
            }
            
            /**
            * \brief Fetches the size of the source.
            * \param source A pointer to the source object.
            * \return The size of the source.
            */
            static unsigned size(U *source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->size();
            }
            
            /**
             * @brief Gets the number of rows of the source.
             * @param source A pointer to the source object.
             * @return The number of rows of the source.
             */
            static unsigned numberOfRows(U *source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->numberOfRows();
            }
            
            /**
             * @brief Gets the number of columns of the source.
             * @param source A pointer to the source object.
             * @return The number of columns of the source.
             */
            static unsigned numberOfColumns(U *source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->numberOfColumns();
            }
            
            /**
             * @brief Gets the NumericalVectorStorage object of the source. Offers utility with respect to the matrix storage
             *        format (e.g., row major, column major, etc.).
             * @param source A pointer to the source object.
             * @return The matrix storage object of the source.
             */
            static shared_ptr<NumericalMatrixStorageDataProvider<T>> getStorageNumericalVectors(U *source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->dataStorage; 
            }
            
            /**
             * @brief Gets the storage type of the source.
             * @param source A pointer to the source object.
             * @return The storage type of the source.
             */
            static NumericalMatrixStorageType getStorageType(U *source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->dataStorage.getStorageType();
            }
        };

        
        
        /// Specialization for NumericalMatrix<U>.
        template<typename U>
        struct dereference_trait<NumericalMatrix<U>> {

            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return A pointer to the data of the source.
            */
            static U* dereference(const NumericalMatrix<U> &source) {
                static_assert(std::is_arithmetic<U>::value, "Template type T must be an arithmetic type (integral or floating-point)");
                return source.getDataPointer();
            }

            /**
            * \brief Fetches the size of the source.
            * \param source A smart pointer to the source object.
            * \return The size of the source.
            */
            static unsigned size(const NumericalMatrix<U> &source) {
                return source.size();
            }
            
            /**
             * @brief Gets the number of rows of the source.
             * @param source A smart pointer to the source object.
             * @return The number of rows of the source.
             */
            static unsigned numberOfRows(const NumericalMatrix<U> &source) {
                return source.numberOfRows();
            }
            
            /**
             * @brief Gets the number of columns of the source.
             * @param source A smart pointer to the source object.
             * @return The number of columns of the source.
             */
            static unsigned numberOfColumns(const NumericalMatrix<U> &source) {
                return source.numberOfColumns();
            }
            
            /**
             * @brief Gets the NumericalVectorStorage object of the source. Offers utility with respect to the matrix storage
             *        format (e.g., row major, column major, etc.).
             * @param source A smart pointer to the source object.
             * @return The matrix storage object of the source.
             */
            static shared_ptr<NumericalMatrixStorageDataProvider<T>> getDataStorageNumericalVectors(const NumericalMatrix<U> &source) {
                return source.dataStorage;
            }
            
            /**
             * @brief Gets the storage type of the source.
             * @param source A smart pointer to the source object.
             * @return The storage type of the source.
             */
            static NumericalMatrixStorageType getStorageType(const NumericalMatrix<U> &source) {
                return source.dataStorage->getStorageType();
            }
        };


        /// Specialization for raw pointer to NumericalMatrix<U>.
        template<typename U>
        struct dereference_trait<NumericalMatrix<U> *> : public dereference_trait_base<NumericalMatrix<U>> {
        };

        /**
        * \brief Base trait for smart pointers.
        *
        * This trait provides a unified way to dereference types like std::unique_ptr and 
        * std::shared_ptr.
        */
        template<template<typename, typename...> class PtrType, typename U>
        struct dereference_trait_pointer_base {
            
            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return A pointer to the data of the source.
            */
            static U *dereference(const PtrType<NumericalMatrix<U>> &source) {
                static_assert(std::is_arithmetic<U>::value, "Template type T must be an arithmetic type (integral or floating-point)");
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->getDataPointer();
            }

            /**
            * \brief Fetches the size of the source.
            * \param source A smart pointer to the source object.
            * \return The size of the source.
            */
            static unsigned size(const PtrType<NumericalMatrix<U>> &source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->size();
            }
            
            /**
             * @brief Gets the number of rows of the source.
             * @param source A smart pointer to the source object.
             * @return The number of rows of the source.
             */
            static unsigned numberOfRows(const PtrType<NumericalMatrix<U>> &source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->numberOfRows();
            }
            
            /**
            * @brief Gets the number of columns of the source.
            * @param source A smart pointer to the source object.
            * @return The number of columns of the source.
            */
            static unsigned numberOfColumns(const PtrType<NumericalMatrix<U>> &source) {
            if (!source) throw std::runtime_error("Null pointer dereferenced");
            return source->numberOfColumns();
            }
            
            /**
             * @brief Gets the NumericalVectorStorage object of the source. Offers utility with respect to the matrix storage
             *        format (e.g., full, CSR, CSC etc.).
             * @param source A smart pointer to the source object.
             * @return The matrix storage object of the source.
             */
            static shared_ptr<NumericalMatrixStorageDataProvider<T>> getDataStorageNumericalVectors(const PtrType<NumericalMatrix<U>> &source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->dataStorage;
            }
            
            /**
             * @brief Gets the storage type of the source.
             * @param source A smart pointer to the source object.
             * @return The storage type of the source.
             */
            static NumericalMatrixStorageType getStorageType(const PtrType<NumericalMatrix<U>> &source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                    return source->dataStorage.getStorageType();
            }
        };

        /// Specialization for std::unique_ptr<NumericalMatrix<U>>.
        template<typename U>
        struct dereference_trait<std::unique_ptr<NumericalMatrix<U>>>
        : public dereference_trait_pointer_base<std::unique_ptr, U> {
        };

        /// Specialization for std::shared_ptr<NumericalMatrix<U>>.
        template<typename U>
        struct dereference_trait<std::shared_ptr<NumericalMatrix<U>>>
        : public dereference_trait_pointer_base<std::shared_ptr, U> {
        };

        /**
        * \brief Trait to standardize dereferencing of various types of numericalVectors.
        *
        * This trait provides a unified way to dereference types such as raw pointers,
        * unique pointers, shared pointers, and direct objects.
        */
        template<typename U>
        struct dereference_trait_vector;

        /**
        * \brief Base trait for raw pointers and direct objects.
        *
        * This trait provides a unified way to dereference types like NumericalVector and 
        * raw pointers to NumericalVector.
        */
        template<typename U>
        struct dereference_trait_vector_base {
            /**
            * \brief Dereferences the source.
            * \param source A pointer to the source object.
            * \return A pointer to the data of the source.
            */
            static U *dereference(U *source) {
                static_assert(std::is_arithmetic<U>::value, "Template type T must be an arithmetic type (integral or floating-point)");

                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->getDataPointer();
            }

            /**
            * \brief Fetches the size of the source.
            * \param source A pointer to the source object.
            * \return The size of the source.
            */
            static unsigned size(U *source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                    return source->size();
            }
        };

        /// Specialization for NumericalVector<U>.
        template<typename U>
        struct dereference_trait_vector<NumericalVector<U>> {

            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return A pointer to the data of the source.
            */
            static U* dereference(const NumericalVector<U> &source) {
                static_assert(std::is_arithmetic<U>::value, "Template type T must be an arithmetic type (integral or floating-point)");
                return source.getDataPointer();
            }

            /**
            * \brief Fetches the size of the source.
            * \param source A smart pointer to the source object.
            * \return The size of the source.
            */
            static unsigned size(const NumericalVector<U> &source) {
                return source.size();
            }
        };


        /// Specialization for raw pointer to NumericalVector<U>.
        template<typename U>
        struct dereference_trait_vector<NumericalVector<U> *> : public dereference_trait_vector_base<NumericalVector<U>> {
        };

        /**
        * \brief Base trait for smart pointers.
        *
        * This trait provides a unified way to dereference types like std::unique_ptr and 
        * std::shared_ptr.
        */
        template<template<typename, typename...> class PtrType, typename U>
        struct dereference_trait_vector_pointer_base {
            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return A pointer to the data of the source.
            */
            static U *dereference(const PtrType<NumericalVector<U>> &source) {
                static_assert(std::is_arithmetic<U>::value, "Template type T must be an arithmetic type (integral or floating-point)");
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->getDataPointer();
            }
            
            /**
            * \brief Fetches the size of the source.
            * \param source A smart pointer to the source object.
            * \return The size of the source.
            */
            static unsigned size(const PtrType<NumericalVector<U>> &source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->size();
            }
        };

        /// Specialization for std::unique_ptr<NumericalVector<U>>.
        template<typename U>
        struct dereference_trait<std::unique_ptr<NumericalVector<U>>>
                : public dereference_trait_pointer_base<std::unique_ptr, U> {
        };

        /// Specialization for std::shared_ptr<NumericalVector<U>>.
        template<typename U>
        struct dereference_trait<std::shared_ptr<NumericalVector<U>>>
                : public dereference_trait_pointer_base<std::shared_ptr, U> {
        };
    };
} // LinearAlgebra

#endif //UNTITLED_NUMERICALMATRIX_H
