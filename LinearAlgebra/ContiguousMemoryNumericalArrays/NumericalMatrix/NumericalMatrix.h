//
// Created by hal9000 on 8/16/23.
//

#ifndef UNTITLED_NUMERICALMATRIX_H
#define UNTITLED_NUMERICALMATRIX_H

#include <vector>
#include <stdexcept>
#include <memory>
#include <type_traits>
#include "../../ParallelizationMethods.h"
#include "../NumericalVector/NumericalVector.h"
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

        shared_ptr<NumericalVector<T>>  _values; ///< Shared ptr to the NumericalVector storing the matrix elements in row-major order.
        
        unsigned int _numberOfRows; ///< Number of rows in the matrix.
        
        unsigned int _numberOfColumns; ///< Number of columns in the matrix.
        
        ParallelizationMethod _parallelizationMethod; ///< Parallelization method used for matrix operations.

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
                return source->data();
            }


            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return The parallelization method of the source.
            */
            static ParallelizationMethod parallelizationMethod(const NumericalMatrix<U> &source) {
                return source->_parallelizationMethod;
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
                return source.data();
            }

            /**
            * \brief Dereferences the source.
            * \param source A smart pointer to the source object.
            * \return The parallelization method of the source.
            */
            static ParallelizationMethod parallelizationMethod(const NumericalMatrix<U> &source) {
                return source._parallelizationMethod;
            }

            /**
            * \brief Fetches the size of the source.
            * \param source A smart pointer to the source object.
            * \return The size of the source.
            */
            static unsigned size(const NumericalMatrix<U> &source) {
                return source.size();
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
                return source->data();
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
