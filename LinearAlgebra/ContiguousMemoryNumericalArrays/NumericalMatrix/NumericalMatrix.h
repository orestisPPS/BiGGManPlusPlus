//
// Created by hal9000 on 8/16/23.
//

#ifndef UNTITLED_NUMERICALMATRIX_H
#define UNTITLED_NUMERICALMATRIX_H

#include "../NumericalVector/NumericalVector.h"
#include "NumericalMatrixStorage/CompressedSparseRowMatrixStorage.h"
#include "NumericalMatrixStorage/FullMatrixStorage.h"
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
                _numberOfRows(rows), _numberOfColumns(columns), _availableThreads(availableThreads), dataStorage(nullptr) {
            
            
            switch (storageType) {
                case FullMatrix:
                    dataStorage = make_unique<FullMatrixStorage<T>>(rows, columns, availableThreads);
                    break;
                case CSR:
                    dataStorage = make_unique<CompressedSparseRowMatrixStorage<T>>(rows, columns, availableThreads);
                    break;
                default:
                    throw std::invalid_argument("Invalid storage type.");
                
            }
        }

        //TODO FIX THIS
        /**
         * @brief Copy constructor for NumericalMatrix.
         * 
         * This copy constructor can handle raw pointers, smart pointers, and other heap objects.
         * 
         * @param other The matrix to be copied from.
         */
        NumericalMatrix(const NumericalMatrix& other) :
                _numberOfRows(other._numberOfRows), _numberOfColumns(other._numberOfColumns), _availableThreads(other._availableThreads) {
                if (dataStorage.getStorageType() == other.dataStorage.getStorageType()){
                    //_storage._values-> = other._values;
                }
                else{
                    //_values = new NumericalVector<T>(other._values->size(), 0, _parallelizationMethod);
                    _deepCopy(other);
                }
        }

        /**
        * @brief Move constructor for NumericalMatrix.
        * 
        * Efficiently transfers the resources of the given matrix to the current matrix.
        * 
        * @param other The matrix to be moved.
        */
        NumericalMatrix(NumericalMatrix&& other) noexcept :
                _numberOfRows(std::exchange(other._numberOfRows, 0)),
                _numberOfColumns(std::exchange(other._numberOfColumns, 0)),
                dataStorage(std::move(other.dataStorage)){
            _availableThreads = other._availableThreads;
        }

        unique_ptr<NumericalMatrixStorage<T>> dataStorage; ///< Storage object for the matrix elements.


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
        * @brief Element access operator for the matrix.
        * 
        * @param row Row index.
        * @param column Column index.
        * @return T& Reference to the element at the specified position.
        */
        T& operator()(unsigned int row, unsigned int column) {
            return dataStorage.getElement(row, column);
        }
        
        const T& operator()(unsigned int row, unsigned int column) const {
            return dataStorage.getElement(row, column);
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
            return _areElementsEqual(other._values);
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
        * @brief Gets the storage type of the matrix.
        * 
        * @return NumericalMatrixStorageType  Enum representing the storage type (FullMatrix, CSR, etc.).
        */
        const NumericalMatrixStorageType& storageType() const {
            return dataStorage->getStorageType();
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
        T* getDataPointer() const {
            return dataStorage->_values->getDataPointer();
        }
        
        /**
        * @brief Gets tha shared pointer to the row major NumericalVector that contains the matrix elements.
        *  
        * @return shared_ptr<NumericalVector<T>> Shared pointer to the data NumericalVector.
        */
        shared_ptr<NumericalVector<T>> getDataNumericalVector() const {
            return dataStorage->_values;
        }

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
/*        template<typename InputType1, typename InputType2>
        void add(const InputType1 &inputMatrix, InputType2 &resultMatrix, T scaleThis = 1, T scaleInput = 1) {

            _checkInputType(inputMatrix);
            _checkInputType(resultMatrix);
            if (numberOfRows() != dereference_trait<InputType1>::numberOfRows(inputMatrix) ||
                numberOfColumns() != dereference_trait<InputType1>::numberOfColumns(inputMatrix)) {
                throw invalid_argument("Vectors must be of the same size.");
            }
            if (numberOfRows() != dereference_trait<InputType2>::numberOfRows(resultMatrix) ||
                numberOfColumns() != dereference_trait<InputType2>::numberOfColumns(resultMatrix)) {
                throw invalid_argument("Vectors must be of the same size.");
            }

            const T *otherData = dereference_trait<InputType1>::dereference(inputMatrix);
            const auto thisData = storage._values->getDataPointer();
            T *resultMatrixData = dereference_trait<InputType2>::dereference(resultMatrix);

            auto addJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < storage._values->size(); ++i) {
                    resultMatrixData[i] = scaleThis * (*thisData)[i] + scaleInput * otherData[i];
                }
            };
            _threading.executeParallelJob(addJob);
        }
        
        *//**
         * @brief Performs element-wise addition of two matrices and stores the resultMatrix in the current matrix.
         * Given two vectors A = [A11, A12, ..., Anm] and B = [B11, B12, ..., Bnm] representing the matrices in row major format,
         * and scalar factors a and b, their addition is:
         * add(A, B) = [a*A11+b*B11, a*A12+b*w12, ..., a*Anm+b*Bnm].
         * 
         * @param inputMatrix The input matrix to add.
         * @param scaleThis Scaling factor for the current vector (default is 1).
         * @param scaleInput Scaling factor for the input vector (default is 1).
         *//*
        template<typename InputType>
        void addIntoThis(const InputType &inputMatrix, T scaleThis = 1, T scaleInput = 1) {
            add(inputMatrix, *this, scaleThis, scaleInput);
        }

        *//**
        * \brief Performs element-wise subtraction of two scaled matrices.
        * Given two vectors A = [A11, A12, ..., Anm] and B = [B11, B12, ..., Bnm] representing the matrices in row major format,
        * and scalar factors a and b, their subtraction is:
        * add(A, B) = [a*A11-b*B11, a*A12-b*B12, ..., a*Anm-b*Bnm].
        * 
        * \param inputMatrix The input matrix to subtract.
        * \param resultMatrix The resultMatrix matrix after subtraction.
        * \param scaleThis Scaling factor for the current vector (default is 1).
        * \param scaleInput Scaling factor for the input vector (default is 1).
        *//*
        template<typename InputType1, typename InputType2>
        void subtract(const InputType1 &inputMatrix, InputType2 &resultMatrix, T scaleThis = 1, T scaleInput = 1) {

            _checkInputType(inputMatrix);
            _checkInputType(resultMatrix);
            if (numberOfRows() != dereference_trait<InputType1>::numberOfRows(inputMatrix) ||
                numberOfColumns() != dereference_trait<InputType1>::numberOfColumns(inputMatrix)) {
                throw invalid_argument("Vectors must be of the same size.");
            }
            if (numberOfRows() != dereference_trait<InputType2>::numberOfRows(resultMatrix) ||
                numberOfColumns() != dereference_trait<InputType2>::numberOfColumns(resultMatrix)) {
                throw invalid_argument("Vectors must be of the same size.");
            }

            const T *otherData = dereference_trait<InputType1>::dereference(inputMatrix);
            const auto thisData = storage._values->getDataPointer();
            T *resultMatrixData = dereference_trait<InputType2>::dereference(resultMatrix);

            auto subtractJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < storage._values->size(); ++i) {
                    resultMatrixData[i] = scaleThis * (*thisData)[i] - scaleInput * otherData[i];
                }
            };
            _threading.executeParallelJob(subtractJob);
        }*/
        
        /**
         * @brief Performs element-wise subtraction of two matrices and stores the result in the current matrix.
         * Given two vectors A = [A11, A12, ..., Anm] and B = [B11, B12, ..., Bnm] representing the matrices in row major format,
         * and scalar factors a and b, their subtraction is:
         * add(A, B) = [a*A11-b*B11, a*A12-b*w12, ..., a*Anm-b*Bnm].
         * 
         * @param inputMatrix The input matrix to subtract.
         * @param scaleThis Scaling factor for the current vector (default is 1).
         * @param scaleInput Scaling factor for the input vector (default is 1).
         */
        template<typename InputType>
        void subtractIntoThis(const InputType &inputMatrix, T scaleThis = 1, T scaleInput = 1) {
            subtract(inputMatrix, *this, scaleThis, scaleInput);
        }
        
    private:
        
        unsigned int _numberOfRows; ///< Number of rows in the matrix.
        
        unsigned int _numberOfColumns; ///< Number of columns in the matrix.
        
        unsigned _availableThreads; ///< Number of available threads for parallelization.
        


/*        *//**
        * @brief Performs a deep copy from the source to the current object.
        * 
        * This method uses the dereference_trait to handle various types of sources 
        * such as raw pointers, unique pointers, shared pointers, and direct objects.
        * 
        * @param source The source object to be copied from.
        *//*
        template<typename InputType>
        void _deepCopy(const InputType &source) {

            if (size() != dereference_trait<InputType>::size(source)) {
                throw std::invalid_argument("Source vector must be the same size as the destination vector.");
            }

            const T *sourceData = dereference_trait<InputType>::dereference(source);

            auto deepCopyThreadJob = [&](unsigned start, unsigned end) {
                for (unsigned i = start; i < end && i < storage._values->size(); ++i) {
                    (*_values)[i] = sourceData[i];
                }
            };
            _threading.executeParallelJob(deepCopyThreadJob, storage._values->size());
        }

        *//**
        * @brief Checks if the elements of the current object are equal to those of the provided source.
        * 
        * This method uses parallelization to perform the comparison and then reduces the results to determine 
        * if all elements are equal.
        * 
        * @param source The source object to be compared with.
        * @return true if all elements are equal, false otherwise.
        *//*
        bool _areElementsEqual(const T *&source, size_t size) {

            if (storage._values->size() != source->size()) {
                throw std::invalid_argument("Source vector must be the same size as the destination vector.");
            }

            auto compareElementsJob = [&](unsigned start, unsigned end) -> bool {
                for (unsigned i = start; i < end && i < storage._values->size(); ++i) {
                    if ((*_values)[i] != source[i]) {
                        return false;
                    }
                }
                return true;
            };

            // Check elements in parallel and reduce the results
            if (_parallelizationMethod == SingleThread) {
                return _threading.executeParallelJobWithReduction(storage._values->size(), compareElementsJob, 1);
            }

            if (_parallelizationMethod == MultiThread) {
                return _threading.executeParallelJobWithReduction(storage._values->size(), compareElementsJob,
                                                        std::thread::hardware_concurrency());
            }
        }*/
        
        void initializeStorage(NumericalMatrixStorageType storageType){
            switch (storageType) {

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
             * @brief Gets the NumericalVectorStorage object of the source. Offers utility with respect to the matrix storage
             *        format (e.g., row major, column major, etc.).
             * @param source A pointer to the source object.
             * @return The matrix storage object of the source.
             */
            static NumericalMatrixStorage<T> getDataStorageNumericalVectors(U *source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->dataStorage;
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
             * @brief Gets the NumericalVectorStorage object of the source. Offers utility with respect to the matrix storage
             *        format (e.g., row major, column major, etc.).
             * @param source A smart pointer to the source object.
             * @return The matrix storage object of the source.
             */
            static NumericalMatrixStorage<T> getDataStorageNumericalVectors(const NumericalMatrix<U> &source) {
                return source.dataStorage;
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
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return The parallelization method of the source.
            */
            static ParallelizationMethod parallelizationMethod(const PtrType<NumericalMatrix<U>> &source) {
                if (!source) throw std::runtime_error("Null pointer dereferenced");
                return source->_parallelizationMethod;
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

        template<typename InputType>
        void _checkInputType(const InputType &input) {
            static_assert(std::is_same<InputType, NumericalMatrix<T>>::value
                                                                     || std::is_same<InputType, std::shared_ptr<NumericalMatrix<T>>>::value
                                                                     || std::is_same<InputType, std::unique_ptr<NumericalMatrix<T>>>::value
                                                                     || std::is_same<InputType, NumericalMatrix<T>*>::value,
                    "Input must be a NumericalMatrix, its pointer, or its smart pointers.");
        }


    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALMATRIX_H
