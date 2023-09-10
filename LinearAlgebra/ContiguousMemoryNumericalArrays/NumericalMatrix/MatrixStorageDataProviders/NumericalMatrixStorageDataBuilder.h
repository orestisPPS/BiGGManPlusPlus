//
// Created by hal9000 on 9/3/23.
//

#ifndef UNTITLED_NUMERICALMATRIXSTORAGEDATABUILDER_H
#define UNTITLED_NUMERICALMATRIXSTORAGEDATABUILDER_H

#include <map>
#include <unordered_map>
#include "../NumericalMatrixEnums.h"
namespace LinearAlgebra{
    
    
    /**
    * @class SparseMatrixBuilder
    * @brief A builder class to currently store a sparse matrix in Coordinate (COO) format and convert it to other s
     * parse matrix formats.
    * 
    * This class provides a foundation to store, access, and modify matrix data during the construction phase.
    * Additionally, it offers utilities to convert the matrix to other sparse matrix representations
    * such as Compressed Sparse Row (CSR), Compressed Sparse Column (CSC) formats and 
    *
    * The COO format is particularly suited for situations where the matrix entries are known one at a time
    * and the matrix is built incrementally. It uses a map to store non-zero elements along with their row 
    * and column indices.
    *
    * @tparam T The datatype of matrix elements. It should support basic arithmetic operations.
    */
    template <typename T>
    class NumericalMatrixStorageDataBuilder{
    public:

        NumericalMatrixStorageDataBuilder(unsigned numberOfRows, unsigned numberOfColumns) :
        _numberOfRows(numberOfRows), _numberOfColumns(numberOfColumns), _elementAssignmentRunning(false),
        _cooMapRowMajor(make_unique<map<tuple<unsigned, unsigned>, T>>()), 
        _cooMapColumnMajor(make_unique<map<tuple<unsigned, unsigned>, T, _compareForColumnMajor>>()) {
            _zero = static_cast<T>(0);  
        }
        

        /**
        * @brief Converts a matrix from Coordinate (COO) format to Compressed Sparse Row (CSR) format.
        * 
        * The CSR format represents a sparse matrix using three one-dimensional arrays:
        * - The values array stores the non-zero elements in row-major order.
        * - The rowOffsets array stores the starting index of the first non-zero element in each row.
        * - The columnIndices array stores the column indices of each element in the values array.
        * 
        * This function converts the matrix from the COO format, stored in the _cooMap, 
        * to the CSR format and returns the three arrays as shared pointers.
        * 
        * @return A tuple containing three shared pointers:
        *         1. A pointer to the values array.
        *         2. A pointer to the columnIndices array.
        *         3. A pointer to the rowOffsets array.
        * 
        * @throws runtime_error if the matrix is empty.
        */
        tuple<shared_ptr<NumericalVector<T>>,
        shared_ptr<NumericalVector<unsigned>>,
        shared_ptr<NumericalVector<unsigned>>>
        getCSRDataVectors() {
            if (_cooMapRowMajor->empty()) {
                throw runtime_error("Matrix is empty.");
            }
            if(!_cooMapColumnMajor->empty())
                throw runtime_error("Column major map is not empty. Error at matrix construction.");
            
            auto values = make_shared<NumericalVector<T>>(_cooMapRowMajor->size());
            auto columnIndices = make_shared<NumericalVector<unsigned>>(_cooMapRowMajor->size());
            auto rowOffsets = make_shared<NumericalVector<unsigned>>(_cooMapRowMajor->size() + 1);

            unsigned currentIndex = 0;
            // Iterate through the entries in the COO map to build the CSR format.
            for (const auto &element: *_cooMapRowMajor) {
                unsigned row = std::get<0>(element.first);
                unsigned col = std::get<1>(element.first);

                (*values)[currentIndex] = element.second;
                (*columnIndices)[currentIndex] = col;

                // Increment the rowOffsets for the current row and all subsequent rows
                for (unsigned r = row + 1; r <= _numberOfRows; r++) {
                    (*rowOffsets)[r]++;
                }

                // Move to the next position in values and columnIndices
                ++currentIndex;
            }
            _cooMapRowMajor->clear();
            return make_tuple(values, columnIndices, rowOffsets);
        }

        /**
        * @brief Converts a matrix from Coordinate (COO) format to Compressed Sparse Column (CSC) format.
        * 
        * The CSC format represents a sparse matrix using three one-dimensional arrays:
        * - The values array stores the non-zero elements in column-major order.
        * - The columnOffsets array stores the starting index of the first non-zero element in each column.
        * - The rowIndices array stores the row indices of each element in the values array.
        * 
        * This function converts the matrix from the COO format, stored in the _cooMap, 
        * to the CSC format and returns the three arrays as shared pointers.
        * 
        * @return A tuple containing three shared pointers:
        *         1. A pointer to the values array.
        *         2. A pointer to the rowIndices array.
        *         3. A pointer to the columnOffsets array.
        * 
        * @throws runtime_error if the matrix is empty.
        */
        tuple<shared_ptr<NumericalVector<T>>,
        shared_ptr<NumericalVector<unsigned>>,
        shared_ptr<NumericalVector<unsigned>>>
        getCSCDataVectors() {
            if (_cooMapColumnMajor->empty()) {
                throw runtime_error("Matrix is empty.");
            }
            if(!_cooMapRowMajor->empty()) {
                throw runtime_error("Row major map is not empty. Error at matrix construction.");
            }

            auto values = make_shared<NumericalVector<T>>(_cooMapColumnMajor->size());
            auto rowIndices = make_shared<NumericalVector<unsigned>>(_cooMapColumnMajor->size());
            auto columnOffsets = make_shared<NumericalVector<unsigned>>(_numberOfColumns + 1, 0); // Initialized with zeros

            unsigned currentIndex = 0;
            unsigned currentColumn = 0;

            for (const auto &element: *_cooMapColumnMajor) {
                unsigned row = std::get<0>(element.first);
                unsigned col = std::get<1>(element.first);

                while (currentColumn < col) {
                    (*columnOffsets)[++currentColumn] = currentIndex;
                }

                (*values)[currentIndex] = element.second;
                (*rowIndices)[currentIndex] = row;

                ++currentIndex;
            }
            while (currentColumn <= _numberOfColumns) {
                (*columnOffsets)[++currentColumn] = currentIndex;
            }

            _cooMapColumnMajor->clear();
            return make_tuple(values, rowIndices, columnOffsets);
        }

        /**
         * @brief Converts a matrix from Compressed Sparse Row (CSR) format to Coordinate (COO) format.
         * 
         * The COO format represents a sparse matrix as a map with row and column indices as keys and matrix values as values.
         * This function converts the matrix from the CSR format to the COO format and stores it in the _cooMap member.
         * 
         * @param values A NumericalVector containing the non-zero values of the matrix in row-major order.
         * @param rowOffsets A NumericalVector containing the starting indices in the 'values' and 'columnIndices' arrays for each row.
         * @param columnIndices A NumericalVector containing the column indices for each value in the 'values' array.
         */
        void getCOOMapFromCSR(NumericalVector<T> &values,
                              NumericalVector<unsigned> &rowOffsets,
                              NumericalVector<unsigned> &columnIndices) {

            // Ensure the provided vectors have valid data
            if (values.empty() || rowOffsets.empty() || columnIndices.empty()) {
                throw runtime_error("CSR data is incomplete.");
            }
            if (_elementAssignmentRunning){
                throw runtime_error("Element assignment is still running. Call finalizeElementAssignment() first.");
            }
            
            // Clear the existing COO map.
            _cooMapRowMajor->clear();

            // Iterate through each row of the matrix.
            for (unsigned row = 0; row < _numberOfRows; ++row) {

                // Get the start and end indices for the current row from the rowOffsets array.
                unsigned startId = rowOffsets[row];
                unsigned endId = rowOffsets[row + 1];

                // Iterate through the non-zero entries in the current row.
                for (unsigned id = startId; id < endId; ++id) {

                    // Get the column index and value for the current non-zero entry.
                    unsigned col = columnIndices[id];
                    T value = values[id];

                    // Insert the non-zero entry into the COO map.
                    (*_cooMapRowMajor)[std::make_tuple(row, col)] = value;
                }
            }
        }
        
        /**
         * @brief Converts a matrix from Compressed Sparse Column (CSC) format to Coordinate (COO) format.
         * 
         * The COO format represents a sparse matrix as a map with row and column indices as keys and matrix values as values.
         * This function converts the matrix from the CSC format to the COO format and stores it in the _cooMap member.
         * 
         * @param values A NumericalVector containing the non-zero values of the matrix in column-major order.
         * @param columnOffsets A NumericalVector containing the starting indices in the 'values' and 'rowIndices' arrays for each column.
         * @param rowIndices A NumericalVector containing the row indices for each value in the 'values' array.
         */
        void getCOOMapFromCSC(NumericalVector<T> &values,
                              NumericalVector<unsigned> &columnOffsets,
                              NumericalVector<unsigned> &rowIndices) {
            if (values.empty() || columnOffsets.empty() || rowIndices.empty()) {
                throw runtime_error("CSC data is incomplete.");
            }

            _cooMapColumnMajor->clear();

            for (unsigned col = 0; col < _numberOfColumns; ++col) {
                for (unsigned id = columnOffsets[col]; id < columnOffsets[col + 1]; ++id) {
                    unsigned row = rowIndices[id];
                    T value = values[id];
                    (*_cooMapColumnMajor)[std::make_tuple(row, col)] = value;
                }
            }
        }

        /**
        * @brief Inserts a value into the matrix at the specified row and column.
        * 
        * @param row The row index.
        * @param column The column index.
        * @param value The value to be inserted.
        * 
        * @throws out_of_range If the row or column index is out of the matrix's range.
         * @throws runtime_error If element assignment is still running.
        */
        void insertElement(unsigned row, unsigned column, const T &value, MatrixElementsOrder order = RowMajor, NumericalMatrixFormType formType = General) {
            if (!_elementAssignmentRunning){
                throw runtime_error("Element assignment is not running. Call enableElementAssignment() first.");
            }
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            if (order == RowMajor){
                _cooMapRowMajor->insert({{row, column}, value});
            }
            else{
                _cooMapColumnMajor->insert({{row, column}, value});
            }
        }

        /**
        * @brief Retrieves a reference to the value at the specified row and column.
        * 
        * @param row The row index.
        * @param column The column index.
        * 
        * @return The value at the specified row and column. Returns 0 if the element is not in the map.
        * 
        * @throws out_of_range If the row or column index is out of the matrix's range.
        */
        T& getElement(unsigned row, unsigned column, MatrixElementsOrder order = RowMajor) {
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            if (order == RowMajor){
                if (_cooMapRowMajor->find({row, column}) == _cooMapRowMajor->end()) {
                    return _zero;
                }
                return _cooMapRowMajor->at({row, column});
            }
            else{
                if (_cooMapColumnMajor->find({row, column}) == _cooMapColumnMajor->end()) {
                    return _zero;
                }
                return _cooMapColumnMajor->at({row, column});
            }
        }
        
        /**
         * @brief Retrieves a constant reference to the value at the specified row and column.
         * @param row The row index.
         * @param column The column index.
         * @return The value at the specified row and column. Returns 0 if the element is not in the map.
         */
        const T& getElement(unsigned row, unsigned column, MatrixElementsOrder order = RowMajor ) const {
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            if (order == RowMajor){
                if (_cooMapRowMajor->find({row, column}) == _cooMapRowMajor->end()) {
                    return _zero;
                }
                return _cooMapRowMajor->at({row, column});
            }
            else{
                if (_cooMapColumnMajor->find({row, column}) == _cooMapColumnMajor->end()) {
                    return _zero;
                }
                return _cooMapColumnMajor->at({row, column});
            }
        }
 
        /**
        * @brief Removes the value at the specified row and column.
        * 
        * @param row The row index.
        * @param column The column index.
        * 
        * @throws out_of_range If the row or column index is out of the matrix's range.
        * @throws runtime_error If element assignment is not running.
        */
        void removeElement(unsigned row, unsigned column, MatrixElementsOrder order = RowMajor) {
            if (!_elementAssignmentRunning){
                throw runtime_error("Element assignment is not running. Call enableElementAssignment() first.");
            }
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            if (order == RowMajor){
                _cooMapRowMajor->erase({row, column});
            }
            else{
                _cooMapColumnMajor->erase({row, column});
            }
        }
        
        /**
         * @brief Enables element assignment and matrix manipulation. If not called, the matrix is read-only.
         */
        void enableElementAssignment(){
            _elementAssignmentRunning = true;
        }
        
        /**
        * @brief Disables element assignment and matrix manipulation. If not called, various sparse format data vectors
        * cannot be retrieved.
        */
        void disableElementAssignment(){
            _elementAssignmentRunning = false;
        }

    private:
        
        unique_ptr<map<tuple<unsigned, unsigned>, T>> _cooMapRowMajor;

        struct _compareForColumnMajor {
            bool operator()(const tuple<unsigned, unsigned>& a, const tuple<unsigned, unsigned>& b) const {
                if (std::get<1>(a) == std::get<1>(b))
                    return std::get<0>(a) < std::get<0>(b);
                return std::get<1>(a) < std::get<1>(b);
            }
        };
        
        unique_ptr<map<tuple<unsigned, unsigned>, T, _compareForColumnMajor>> _cooMapColumnMajor;        
        
        unsigned _numberOfRows;
        
        unsigned _numberOfColumns;
        
        bool _elementAssignmentRunning;
        
        T _zero;



 
    };
}

#endif //UNTITLED_NUMERICALMATRIXSTORAGEDATABUILDER_H
