//
// Created by hal9000 on 9/3/23.
//

#ifndef UNTITLED_SPARSEMATRIXSTORAGE_H
#define UNTITLED_SPARSEMATRIXSTORAGE_H

#include <map>
#include <unordered_map>
#include "NumericalMatrixStorage.h"

namespace LinearAlgebra{
    /**
     * @brief A class that provides storage for sparse matrices in COO format using a hash map. Can be used as it is or as a base class.
     * 
     * This class derives from `NumericalMatrixStorage` and provides a COO (Coordinate) storage format
     * for sparse matrices, using an `unordered_map` with (row, column) pairs as keys and values as data.
     *
     * @tparam T The type of elements stored in the matrix (e.g., `double`, `float`).
     */
    template <typename T>
    class SparseMatrixStorage : public NumericalMatrixStorage<T> {
    public:
        /**
         * @brief Constructs a new SparseMatrixStorage object.
         * 
         * @param storageType The storage type for the matrix.
         * @param numberOfRows The number of rows in the matrix.
         * @param numberOfColumns The number of columns in the matrix.
         * @param parallelizationMethod The parallelization method used for matrix operations.
         */
        SparseMatrixStorage(NumericalMatrixStorageType storageType, unsigned numberOfRows, unsigned numberOfColumns, ParallelizationMethod parallelizationMethod) :
                NumericalMatrixStorage<T>(storageType, numberOfRows, numberOfColumns, parallelizationMethod) {
            _cooHashMap = make_unique<unordered_map<pair<unsigned, unsigned>, unique_ptr<T>>>();
        }

        /**
        * @brief Destroys the SparseMatrixStorage object and clears the hash map.
        */
        ~SparseMatrixStorage() {
            _cooHashMap->clear();
        }
        
        T& getElement(unsigned row, unsigned column) override {
            return _get(row, column);
        }


    protected:
        
        /// The hash map used to store the matrix elements in COO format.
        unique_ptr<unordered_map<std::pair<unsigned, unsigned>, unique_ptr<T>>> _cooHashMap;
        
        /**
         * @brief Finalizes the element assignment process and creates the values vector.
         * 
         * @param clearHashMap If true, the hash map is cleared after the values vector is created. Default is true.
         */
        void _finalizeSparseElementAssignment(bool clearHashMap = true) {
            if (!this->_elementAssignmentRunning){
                throw runtime_error("Element assignment is not running. Call initializeElementAssignment() first.");
            }
            this->_elementAssignmentRunning = false;
            this->_values = make_shared<NumericalVector<T>>(_cooHashMap->size(), this->_parallelizationMethod);
            unsigned i = 0;
            auto thisData = this->_values->getData();
            for (auto &element : *_cooHashMap) {
                thisData[i] = element.second;
                ++i;
            }
            if (clearHashMap)
                _cooHashMap->clear();
        }
        
        /**
        * @brief Inserts a value into the matrix at the specified row and column.
        * 
        * @param row The row index.
        * @param column The column index.
        * @param value The value to be inserted.
        * 
        * @throws out_of_range If the row or column index is out of the matrix's range.
        */
        void _insert(unsigned row, unsigned column, const T &value) {
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            _cooHashMap[{row, column}] = value;
        }

        /**
        * @brief Retrieves the value at the specified row and column.
        * 
        * @param row The row index.
        * @param column The column index.
        * 
        * @return The value at the specified row and column. Returns 0 if the element is not in the hash map.
        * 
        * @throws out_of_range If the row or column index is out of the matrix's range.
        */
        T& _get(unsigned row, unsigned column) {
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            auto it = _cooHashMap.find({row, column});
            if (it == _cooHashMap.end()) {
                return 0;
            }
            return it->second;
        }

        /**
        * @brief Removes the value at the specified row and column.
        * 
        * @param row The row index.
        * @param column The column index.
        * 
        * @throws out_of_range If the row or column index is out of the matrix's range.
        */
        void _remove(unsigned row, unsigned column) {
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            _cooHashMap.erase({row, column});
        }


    };
}

#endif //UNTITLED_SPARSEMATRIXSTORAGE_H
