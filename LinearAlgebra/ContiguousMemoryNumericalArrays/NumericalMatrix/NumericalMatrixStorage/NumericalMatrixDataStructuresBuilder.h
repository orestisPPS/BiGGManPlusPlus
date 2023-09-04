//
// Created by hal9000 on 9/3/23.
//

#ifndef UNTITLED_NUMERICALMATRIXDATASTRUCTURESBUILDER_H
#define UNTITLED_NUMERICALMATRIXDATASTRUCTURESBUILDER_H

#include <map>
#include <unordered_map>
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
    class NumericalMatrixDataStructuresBuilder{
    public:

        NumericalMatrixDataStructuresBuilder(unsigned numberOfRows, unsigned numberOfColumns) :
        _numberOfRows(numberOfRows), _numberOfColumns(numberOfColumns), _elementAssignmentRunning(false),
        _cooMap(make_unique<map<tuple<unsigned, unsigned>, T>>()){
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
        *         2. A pointer to the rowOffsets array.
        *         3. A pointer to the columnIndices array.
        * 
        * @throws runtime_error if the matrix is empty.
        */
        tuple<shared_ptr<NumericalVector<T>>,
        shared_ptr<NumericalVector<unsigned>>,
        shared_ptr<NumericalVector<unsigned>>>
        getCSRDataVectors() {
            auto values = make_shared<NumericalVector<T>>(_cooMap->size());
            auto rowOffsets = make_shared<NumericalVector<unsigned>>(_cooMap->size() + 1);
            auto columnIndices = make_shared<NumericalVector<unsigned>>(_cooMap->size());
            if (_cooMap->empty()) {
                throw runtime_error("Matrix is empty.");
            }
            if (_elementAssignmentRunning){
                throw runtime_error("Element assignment is still running. Call finalizeElementAssignment() first.");
            }
            else{
                unsigned currentRow = 0;
                unsigned currentIndex = 0;
                (*rowOffsets)[0] = 0;

                // Iterate through the entries in the COO map (sorted by row, then column) to build the CSR format.
                for (const auto &element: *_cooMap) {

                    unsigned row = std::get<0>(element.first);
                    unsigned col = std::get<1>(element.first);

                    (*values)[currentIndex] = element.second;
                    (*columnIndices)[currentIndex] = col;

                    // Move to the next position in values and columnIndices.
                    ++currentIndex;

                    // If there are any rows from 'currentRow' to 'row' that don't have any non-zero elements,
                    // their starting and ending position would be the same in the 'values' vector.
                    while (currentRow < row) {
                        ++currentRow; // Move to the next row.

                        // Update the 'rowOffsets' vector to point to the current position in 'values'.
                        (*rowOffsets)[currentRow] = currentIndex;
                    }
                }

                // After processing all entries from the COO map, fill in the remaining offsets.
                // This is necessary for rows at the end of the matrix that have no non-zero elements.
                while (currentRow < _numberOfRows) {
                    ++currentRow; // Move to the next row.

                    // Update the 'rowOffsets' vector to point to the end of the 'values' vector.
                    (*rowOffsets)[currentRow] = currentIndex;
                }

                return make_tuple(values, rowOffsets, columnIndices);
            }
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
            auto values = make_shared<NumericalVector<T>>(_cooMap->size());
            auto columnOffsets = make_shared<NumericalVector<unsigned>>(_cooMap->size() + 1);
            auto rowIndices = make_shared<NumericalVector<unsigned>>(_cooMap->size());

            if (_cooMap->empty()) {
                throw runtime_error("Matrix is empty.");
            }
            if (_elementAssignmentRunning){
                throw runtime_error("Element assignment is still running. Call finalizeElementAssignment() first.");
            }
            else{
                unsigned currentColumn = 0;
                unsigned currentIndex = 0;
                (*columnOffsets)[0] = 0;

                // Iterate through the entries in the COO map (sorted by column, then row) to build the CSC format.
                for (const auto &element: *_cooMap) {
                    unsigned row = std::get<0>(element.first);
                    unsigned col = std::get<1>(element.first);

                    (*values)[currentIndex] = element.second;
                    (*rowIndices)[currentIndex] = row;

                    // Move to the next position in values and rowIndices.
                    ++currentIndex;

                    while (currentColumn < col) {
                        ++currentColumn; // Move to the next column.

                        // Update the 'columnOffsets' vector to point to the current position in 'values'.
                        (*columnOffsets)[currentColumn] = currentIndex;
                    }
                }

                // After processing all entries from the COO map, fill in the remaining offsets.
                while (currentColumn < _numberOfColumns) {
                    ++currentColumn; // Move to the next column.

                    // Update the 'columnOffsets' vector to point to the end of the 'values' vector.
                    (*columnOffsets)[currentColumn] = currentIndex;
                }
            }
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

            // Clear the existing COO map.
            _cooMap->clear();

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
                    (*_cooMap)[std::make_tuple(row, col)] = value;
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

            // Ensure the provided vectors have valid data
            if (values.empty() || columnOffsets.empty() || rowIndices.empty()) {
                throw runtime_error("CSC data is incomplete.");
            }

            // Clear the existing COO map.
            _cooMap->clear();

            // Iterate through each column of the matrix.
            for (unsigned col = 0; col < _numberOfColumns; ++col) {

                // Get the start and end indices for the current column from the columnOffsets array.
                unsigned startId = columnOffsets[col];
                unsigned endId = columnOffsets[col + 1];

                // Iterate through the non-zero entries in the current column.
                for (unsigned id = startId; id < endId; ++id) {

                    // Get the row index and value for the current non-zero entry.
                    unsigned row = rowIndices[id];
                    T value = values[id];

                    // Insert the non-zero entry into the COO map.
                    (*_cooMap)[std::make_tuple(row, col)] = value;
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
        void insertElement(unsigned row, unsigned column, const T &value) {
            if (!_elementAssignmentRunning){
                throw runtime_error("Element assignment is not running. Call enableElementAssignment() first.");
            }
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            _cooMap->insert(pair<tuple<unsigned, unsigned>, T>({row, column}, value));
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
        T& getElement(unsigned row, unsigned column) {
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            if (_cooMap->find({row, column}) == _cooMap->end()) {
                return _zero;
            }
            return _cooMap->at({row, column});
        }
        
        /**
         * @brief Retrieves a constant reference to the value at the specified row and column.
         * @param row The row index.
         * @param column The column index.
         * @return The value at the specified row and column. Returns 0 if the element is not in the map.
         */
        const T& getElement(unsigned row, unsigned column) const {
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            if (_cooMap->find({row, column}) == _cooMap->end()) {
                return 0;
            }
            return _cooMap->at({row, column});
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
        void removeElement(unsigned row, unsigned column) {
            if (!_elementAssignmentRunning){
                throw runtime_error("Element assignment is not running. Call enableElementAssignment() first.");
            }
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns) {
                throw out_of_range("Row or column index out of range.");
            }
            _cooMap->erase({row, column});
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
        
        unique_ptr<map<tuple<unsigned, unsigned>, T>> _cooMap;
        
        unsigned _numberOfRows;
        
        unsigned _numberOfColumns;
        
        bool _elementAssignmentRunning;
        
        T _zero;

 
    };
}

#endif //UNTITLED_NUMERICALMATRIXDATASTRUCTURESBUILDER_H
