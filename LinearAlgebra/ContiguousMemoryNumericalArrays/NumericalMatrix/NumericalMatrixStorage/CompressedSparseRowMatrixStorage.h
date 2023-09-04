//
// Created by hal9000 on 9/2/23.
//

#ifndef UNTITLED_COMPRESSEDSPARSEROWMATRIXSTORAGE_H
#define UNTITLED_COMPRESSEDSPARSEROWMATRIXSTORAGE_H

#include "SparseMatrixStorage.h"

namespace LinearAlgebra {
    template <typename T>
    class CompressedSparseRowMatrixStorage : public SparseMatrixStorage<T>{
    public:
        explicit CompressedSparseRowMatrixStorage(unsigned numberOfRows, unsigned numberOfColumns, unsigned numberOfThreads)
                : SparseMatrixStorage<T>(numberOfRows, numberOfColumns, numberOfThreads){
            this->_storageType = NumericalMatrixStorageType::CSR;
            this->_values = make_shared<NumericalVector<T>>(0, 0, numberOfThreads);
            _columnIndices = make_shared<NumericalVector<unsigned>>(0, 0, numberOfThreads);
            _rowOffsets = make_shared<NumericalVector<unsigned>>(numberOfRows + 1, 0, numberOfThreads);
            (*_rowOffsets)[0] = 0;
        }

        vector<shared_ptr<NumericalVector<unsigned>>> getSupplementaryVectors() override{
            return {this->_columnIndices, this->_rowOffsets};
        }

        T& getElement(unsigned int row, unsigned int column) override {
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns)
                throw runtime_error("Row or column index out of bounds.");

            if (this->_elementAssignmentRunning) {
                return this->_builder.getElement(row, column);
            }
            else {
                unsigned rowStart = (*_rowOffsets)[row];
                unsigned rowEnd = (*_rowOffsets)[row + 1];
                for (unsigned i = rowStart; i < rowEnd; i++) {
                    if ((*_columnIndices)[i] == column)
                        return (*this->_values)[i];
                }
                return this->_zero; // Return zero of type T
            }
        }

        /**
        * @brief Sets the value at the specified row and column index of the sparse matrix.
        * 
        * The function is designed to handle both the case when the matrix is being built (using the COO format in the builder)
        * and when the matrix is already complete (using the CSR format).
         * WARNING : Avoid calling this function when element assignment is not running. When a lot of elements need to be
         * changed and stored in the matrix, it is more efficient to use the builder to insert the elements and then call
         * finalizeElementAssignment() to convert the matrix to CSR format.
        *
        * @param row The row index at which to set the value.
        * @param column The column index at which to set the value.
        * @param value The value to be set.
        * 
        * @throws runtime_error if the row or column indices are out of bounds.
        * 
        * @note 
        * 1. If the element is already present in the CSR format, the function updates the value.
        * 2. If the matrix is still being built (i.e., `_elementAssignmentRunning` is true), the function inserts the value in the COO format using the builder.
        * 3. If the matrix is in CSR format and the element doesn't exist, the function inserts the element in the correct position in the CSR format.
        *    However, if the value to be set is zero, the insertion is skipped since the CSR format doesn't store zero values.
        * 4. The insertion in the CSR format involves:
        *    - Resizing the `values` and `columnIndices` vectors (size + 1).
        *    - Shifting the existing elements to accommodate the new value.
        *    - Adjusting the row offsets for the subsequent rows.
        */
        void setElement(unsigned int row, unsigned int column, T value) override {
            if (row >= this->_numberOfRows || column >= this->_numberOfColumns)
                throw runtime_error("Row or column index out of bounds.");

            bool elementFound = false;
            if (this->_elementAssignmentRunning) {
                this->_builder.insertElement(row, column, value);
                elementFound = true;
                return;
            }
            
            //Check if the element already exists in csr format
            unsigned rowStart = (*_rowOffsets)[row];
            unsigned rowEnd = (*_rowOffsets)[row + 1];
            for (unsigned i = rowStart; i < rowEnd; i++) {
                if ((*_columnIndices)[i] == column) {
                    (*this->_values)[i] = value;
                    elementFound = true;
                    break;
                }
            }
            if (!elementFound and value != static_cast<T>(0)){
                vector<T>& valuesData = *this->_values->getData();
                vector<unsigned>& columnIndicesData = *this->_columnIndices->getData();
                vector<unsigned>& rowOffsetsData = *this->_rowOffsets->getData();

                // Resize the vectors to accommodate the new element
                valuesData.resize(valuesData.size() + 1);
                columnIndicesData.resize(columnIndicesData.size() + 1);

                // Shift elements to the right by one position, starting from the end
                for (unsigned i = valuesData.size() - 1; i > rowEnd; i--) {
                    valuesData[i] = valuesData[i-1];
                    columnIndicesData[i] = columnIndicesData[i-1];
                }

                // Insert the new value and column index at the identified location
                valuesData[rowEnd] = value;
                columnIndicesData[rowEnd] = column;

                // Adjust the row offsets for subsequent rows
                for (unsigned i = row + 1; i <= this->_numberOfRows; i++) {
                    rowOffsetsData[i]++;
                }

            }
        }

        void eraseElement(unsigned int row, unsigned int column) override {

            if (row >= this->_numberOfRows || column >= this->_numberOfColumns)
                throw runtime_error("Row or column index out of bounds.");

            if (this->_elementAssignmentRunning) {
                this->_builder.removeElement(row, column);
                return;
            }

            // Check if the element exists in csr format
            unsigned rowStart = (*_rowOffsets)[row];
            unsigned rowEnd = (*_rowOffsets)[row + 1];

            // Identify the position of the element to be erased
            unsigned positionToErase = rowEnd; // Default to end (i.e., element not found)
            for (unsigned i = rowStart; i < rowEnd; i++) {
                if ((*_columnIndices)[i] == column) {
                    positionToErase = i;
                    break;
                }
            }

            // If the element is found
            if (positionToErase != rowEnd) {
                auto &valuesData = *this->_values->getData();
                auto &columnIndicesData = *this->_columnIndices->getData();
                auto &rowOffsetsData = *this->_rowOffsets->getData();

                // Shift elements to the left by one position, starting from the identified position
                for (unsigned i = positionToErase; i < valuesData.size() - 1; i++) {
                    valuesData[i] = valuesData[i+1];
                    columnIndicesData[i] = columnIndicesData[i+1];
                }

                // Resize the vectors to remove the last element
                valuesData.resize(valuesData.size() - 1);
                columnIndicesData.resize(columnIndicesData.size() - 1);

                // Adjust the row offsets for subsequent rows
                for (unsigned i = row + 1; i <= this->_numberOfRows; i++) {
                    rowOffsetsData[i]--;
                }
            }
        }

        shared_ptr<NumericalVector<T>> getRowSharedPtr(unsigned row) override {
            if (row >= this->_numberOfRows) {
                throw runtime_error("Row index out of bounds.");
            }
            auto rowVector = make_shared<NumericalVector<T>>(this->_numberOfColumns, static_cast<T>(0));
            unsigned int rowStart = (*_rowOffsets)[row];
            unsigned int rowEnd = (*_rowOffsets)[row + 1];

            for (unsigned int i = rowStart; i < rowEnd; i++) {
                (*rowVector)[(*_columnIndices)[i]] = (*this->_values)[i];
            }
            return rowVector;
        }

        shared_ptr<NumericalVector<T>> getColumnSharedPtr(unsigned column) override {
            if (column >= this->_numberOfColumns) {
                throw runtime_error("Column index out of bounds.");
            }
            auto columnVector = make_shared<NumericalVector<T>>(this->_numberOfRows, static_cast<T>(0));
            for (unsigned int row = 0; row < this->_numberOfRows; row++) {
                unsigned int rowStart = (*_rowOffsets)[row];
                unsigned int rowEnd = (*_rowOffsets)[row + 1];
                for (unsigned int i = rowStart; i < rowEnd; i++) {
                    if ((*_columnIndices)[i] == column) {
                        (*columnVector)[row] = (*this->_values)[i];
                        break;
                    }
                }
            }
        }

        void initializeElementAssignment() override {
            if (this->_elementAssignmentRunning)
                throw runtime_error("Element assignment is already running. Call finalizeElementAssignment() first.");

            this->_elementAssignmentRunning = true;
            this->_builder.enableElementAssignment();
        }

        void finalizeElementAssignment() override {
            if (!this->_elementAssignmentRunning) {
                throw runtime_error("Element assignment is not running. Call initializeElementAssignment() first.");
            }
            this->_elementAssignmentRunning = false;
            this->_builder.disableElementAssignment();
            auto dataVectors = this->_builder.getCSRDataVectors();
            
            this->_values = std::move(get<0>(dataVectors));
            this->_columnIndices = std::move(get<1>(dataVectors));
            this->_rowOffsets = std::move(get<2>(dataVectors));
        }

        /*void matrixAdd(NumericalMatrixStorage<T> &inputMatrixData,
                       NumericalMatrixStorage<T> &resultMatrixData,
                       T scaleThis, T scaleOther) override {

            vector<T>& thisValues = *this->_values->getData();
            vector<unsigned>& thisColumnIndices = *this->_columnIndices->getData();
            vector<unsigned>& thisRowPointers = *this->_rowOffsets->getData();

            auto inputStorage = inputMatrixData.getNecessaryStorageVectors();
            vector<T>& inputValues = inputStorage[0];
            vector<unsigned>& inputColumnIndices = inputStorage[1];
            vector<unsigned>& inputRowPointers = inputStorage[2];

            auto resultStorage = resultMatrixData[0].getNecessaryStorageVectors();
            vector<T>& resultValues = resultStorage[0];
            vector<unsigned>& resultColumnIndices = resultStorage[1];
            vector<unsigned>& resultRowPointers = resultStorage[2];

            unsigned numThreads = this->_threading.getAvailableThreads();
            vector<vector<T>> localResultsValues(numThreads);
            vector<vector<unsigned>> localResultsColumnIndices(numThreads);
            vector<vector<unsigned>> localResultsRowPointers(numThreads);

            auto matrixAddJob = [&](unsigned startRow, unsigned endRow, unsigned threadId) -> void {
                for (unsigned row = startRow; row < endRow; ++row) {
                    unsigned thisRowStart = thisRowPointers[row];
                    unsigned thisRowEnd = thisRowPointers[row + 1];
                    unsigned inputRowStart = inputRowPointers[row];
                    unsigned inputRowEnd = inputRowPointers[row + 1];

                    unsigned i = thisRowStart, j = inputRowStart;
                    while (i < thisRowEnd && j < inputRowEnd) {
                        if (thisColumnIndices[i] == inputColumnIndices[j]) {
                            localResultsValues[threadId].push_back(scaleThis * thisValues[i] + scaleOther * inputValues[j]);
                            localResultsColumnIndices[threadId].push_back(thisColumnIndices[i]);
                            i++;
                            j++;
                        } else if (thisColumnIndices[i] < inputColumnIndices[j]) {
                            localResultsValues[threadId].push_back(scaleThis * thisValues[i]);
                            localResultsColumnIndices[threadId].push_back(thisColumnIndices[i]);
                            i++;
                        } else {
                            localResultsValues[threadId].push_back(scaleOther * inputValues[j]);
                            localResultsColumnIndices[threadId].push_back(inputColumnIndices[j]);
                            j++;
                        }
                    }
                    while (i < thisRowEnd) {
                        localResultsValues[threadId].push_back(scaleThis * thisValues[i]);
                        localResultsColumnIndices[threadId].push_back(thisColumnIndices[i]);
                        i++;
                    }
                    while (j < inputRowEnd) {
                        localResultsValues[threadId].push_back(scaleOther * inputValues[j]);
                        localResultsColumnIndices[threadId].push_back(inputColumnIndices[j]);
                        j++;
                    }
                    localResultsRowPointers[threadId].push_back(localResultsValues[threadId].size());
                }
            };
            
            this->_threading.executeParallelJob(matrixAddJob, this->_numberOfRows);

            // Combine the local results from all threads into the main result storage
            for (unsigned int t = 0; t < numThreads; ++t) {
                resultValues.insert(resultValues.end(), localResultsValues[t].begin(), localResultsValues[t].end());
                resultColumnIndices.insert(resultColumnIndices.end(), localResultsColumnIndices[t].begin(), localResultsColumnIndices[t].end());
                if (t > 0) {
                    // Adjust the row pointers for the threads after the first one
                    unsigned offset = localResultsRowPointers[t-1].back();
                    for (unsigned val : localResultsRowPointers[t]) {
                        resultRowPointers.push_back(val + offset);
                    }
                } else {
                    resultRowPointers.insert(resultRowPointers.end(), localResultsRowPointers[t].begin(), localResultsRowPointers[t].end());
                }
            }
        }

        
        void matrixAddIntoThis(NumericalMatrixStorage<T>&inputMatrixData, T scaleThis, T scaleOther) override {

            vector<T>& thisValues = *this->_values->getData();
            vector<unsigned>& thisColumnIndices = *this->_columnIndices->getData();
            vector<unsigned>& thisRowPointers = *this->_rowOffsets->getData();

            auto inputStorage = inputMatrixData[0].getNecessaryStorageVectors();
            vector<T>& inputValues = inputStorage[0];
            vector<unsigned>& inputColumnIndices = inputStorage[1];
            vector<unsigned>& inputRowPointers = inputStorage[2];


            unsigned numThreads = this->_threading.getAvailableThreads();
            vector<vector<T>> localResultsValues(numThreads);
            vector<vector<unsigned>> localResultsColumnIndices(numThreads);
            vector<vector<unsigned>> localResultsRowPointers(numThreads);
            
            auto matrixAddJob = [&](unsigned startRow, unsigned endRow, unsigned threadId) -> void {
                for (unsigned row = startRow; row < endRow; ++row) {
                    unsigned thisRowStart = thisRowPointers[row];
                    unsigned thisRowEnd = thisRowPointers[row + 1];
                    unsigned inputRowStart = inputRowPointers[row];
                    unsigned inputRowEnd = inputRowPointers[row + 1];

                    unsigned i = thisRowStart, j = inputRowStart;
                    while (i < thisRowEnd && j < inputRowEnd) {
                        if (thisColumnIndices[i] == inputColumnIndices[j]) {
                            thisValues[i] = scaleThis * thisValues[i] + scaleOther * inputValues[j];
                            i++;
                            j++;
                        } else if (thisColumnIndices[i] < inputColumnIndices[j]) {
                            thisValues[i] = scaleThis * thisValues[i];
                            i++;
                        } else {
                            thisValues[i] = scaleOther * inputValues[j];
                            j++;
                        }
                    }
                    while (i < thisRowEnd) {
                        thisValues[i] = scaleThis * thisValues[i];
                        i++;
                    }
                    while (j < inputRowEnd) {
                        thisValues[i] = scaleOther * inputValues[j];
                        j++;
                    }
                }
            };
            
            this->_threading.executeParallelJob(matrixAddJob, this->_numberOfRows);
            
            thisValues.clear();
            thisColumnIndices.clear();
            thisRowPointers.clear();

            // Combine the local results from all threads into the main result storage
            for (unsigned int t = 0; t < numThreads; ++t) {
                this->_values->insert(this->_values->end(), localResultsValues[t].begin(), localResultsValues[t].end());
                this->_columnIndices->insert(this->_columnIndices->end(), localResultsColumnIndices[t].begin(), localResultsColumnIndices[t].end());

                if (t > 0) {
                    // Adjust the row pointers for the threads after the first one
                    unsigned offset = localResultsRowPointers[t-1].back();
                    for (unsigned val : localResultsRowPointers[t]) {
                        this->_rowOffsets->push_back(val + offset);
                    }
                } else {
                    this->_rowOffsets->insert(this->_rowOffsets->end(), localResultsRowPointers[t].begin(), localResultsRowPointers[t].end());
                }
            }
        }
        
        void matrixSubtract(NumericalMatrixStorage<T> &inputMatrixData,
                            NumericalMatrixStorage<T> &resultMatrixData,
                            T scaleThis, T scaleOther) override {

            vector<T>& thisValues = *this->_values->getData();
            vector<unsigned>& thisColumnIndices = *this->_columnIndices->getData();
            vector<unsigned>& thisRowPointers = *this->_rowOffsets->getData();

            auto inputStorage = inputMatrixData[0].getNecessaryStorageVectors();
            vector<T>& inputValues = inputStorage[0];
            vector<unsigned>& inputColumnIndices = inputStorage[1];
            vector<unsigned>& inputRowPointers = inputStorage[2];

            auto resultStorage = resultMatrixData[0].getNecessaryStorageVectors();
            vector<T>& resultValues = resultStorage[0];
            vector<unsigned>& resultColumnIndices = resultStorage[1];
            vector<unsigned>& resultRowPointers = resultStorage[2];

            unsigned numThreads = this->_threading.getAvailableThreads();
            vector<vector<T>> localResultsValues(numThreads);
            vector<vector<unsigned>> localResultsColumnIndices(numThreads);
            vector<vector<unsigned>> localResultsRowPointers(numThreads);

            auto matrixSubtractJob = [&](unsigned startRow, unsigned endRow, unsigned threadId) -> void {
                for (unsigned row = startRow; row < endRow; ++row) {
                    unsigned thisRowStart = thisRowPointers[row];
                    unsigned thisRowEnd = thisRowPointers[row + 1];
                    unsigned inputRowStart = inputRowPointers[row];
                    unsigned inputRowEnd = inputRowPointers[row + 1];

                    unsigned i = thisRowStart, j = inputRowStart;
                    while (i < thisRowEnd && j < inputRowEnd) {
                        if (thisColumnIndices[i] == inputColumnIndices[j]) {
                            localResultsValues[threadId].push_back(scaleThis * thisValues[i] - scaleOther * inputValues[j]);
                            localResultsColumnIndices[threadId].push_back(thisColumnIndices[i]);
                            i++;
                            j++;
                        } else if (thisColumnIndices[i] < inputColumnIndices[j]) {
                            localResultsValues[threadId].push_back(scaleThis * thisValues[i]);
                            localResultsColumnIndices[threadId].push_back(thisColumnIndices[i]);
                            i++;
                        } else {
                            localResultsValues[threadId].push_back(-scaleOther * inputValues[j]);
                            localResultsColumnIndices[threadId].push_back(inputColumnIndices[j]);
                            j++;
                        }
                    }
                    while (i < thisRowEnd) {
                        localResultsValues[threadId].push_back(scaleThis * thisValues[i]);
                        localResultsColumnIndices[threadId].push_back(thisColumnIndices[i]);
                        i++;
                    }
                    while (j < inputRowEnd) {
                        localResultsValues[threadId].push_back(-scaleOther * inputValues[j]);
                        localResultsColumnIndices[threadId].push_back(inputColumnIndices[j]);
                        j++;
                    }
                    localResultsRowPointers[threadId].push_back(localResultsValues[threadId].size());
                }
            };
            
            this->_threading.executeParallelJob(matrixSubtractJob, this->_numberOfRows);
            
            // Combine the local results from all threads into the main result storage
            for (unsigned int t = 0; t < numThreads; ++t) {
                resultValues.insert(resultValues.end(), localResultsValues[t].begin(), localResultsValues[t].end());
                resultColumnIndices.insert(resultColumnIndices.end(), localResultsColumnIndices[t].begin(), localResultsColumnIndices[t].end());
                if (t > 0) {
                    // Adjust the row pointers for the threads after the first one
                    unsigned offset = localResultsRowPointers[t-1].back();
                    for (unsigned val : localResultsRowPointers[t]) {
                        resultRowPointers.push_back(val + offset);
                    }
                } else {
                    resultRowPointers.insert(resultRowPointers.end(), localResultsRowPointers[t].begin(), localResultsRowPointers[t].end());
                }
            }
        }
        
        void matrixSubtractIntoThis(NumericalMatrixStorage<T> &inputMatrixData, T scaleThis, T scaleOther) override {

            vector<T>& thisValues = *this->_values->getData();
            vector<unsigned>& thisColumnIndices = *this->_columnIndices->getData();
            vector<unsigned>& thisRowPointers = *this->_rowOffsets->getData();

            auto inputStorage = inputMatrixData[0].getNecessaryStorageVectors();
            vector<T>& inputValues = inputStorage[0];
            vector<unsigned>& inputColumnIndices = inputStorage[1];
            vector<unsigned>& inputRowPointers = inputStorage[2];

            unsigned numThreads = this->_threading.getAvailableThreads();
            vector<vector<T>> localResultsValues(numThreads);
            vector<vector<unsigned>> localResultsColumnIndices(numThreads);
            vector<vector<unsigned>> localResultsRowPointers(numThreads);

            auto matrixSubtractJob = [&](unsigned startRow, unsigned endRow, unsigned threadId) -> void {
                for (unsigned row = startRow; row < endRow; ++row) {
                    unsigned thisRowStart = thisRowPointers[row];
                    unsigned thisRowEnd = thisRowPointers[row + 1];
                    unsigned inputRowStart = inputRowPointers[row];
                    unsigned inputRowEnd = inputRowPointers[row + 1];

                    unsigned i = thisRowStart, j = inputRowStart;
                    while (i < thisRowEnd && j < inputRowEnd) {
                        if (thisColumnIndices[i] == inputColumnIndices[j]) {
                            thisValues[i] = scaleThis * thisValues[i] - scaleOther * inputValues[j];
                            i++;
                            j++;
                        } else if (thisColumnIndices[i] < inputColumnIndices[j]) {
                            thisValues[i] = scaleThis * thisValues[i];
                            i++;
                        } else {
                            thisValues[i] = -scaleOther * inputValues[j];
                            j++;
                        }
                    }
                    while (i < thisRowEnd) {
                        thisValues[i] = scaleThis * thisValues[i];
                        i++;
                    }
                    while (j < inputRowEnd) {
                        thisValues[i] = -scaleOther * inputValues[j];
                        j++;
                    }
                }
            };
            
            this->_threading.executeParallelJob(matrixSubtractJob, this->_numberOfRows);
            
            thisValues.clear();
            thisColumnIndices.clear();
            thisRowPointers.clear();
            
            // Combine the local results from all threads into the main result storage
            for (unsigned int t = 0; t < numThreads; ++t) {
                this->_values->insert(this->_values->end(), localResultsValues[t].begin(), localResultsValues[t].end());
                this->_columnIndices->insert(this->_columnIndices->end(), localResultsColumnIndices[t].begin(), localResultsColumnIndices[t].end());

                if (t > 0) {
                    // Adjust the row pointers for the threads after the first one
                    unsigned offset = localResultsRowPointers[t-1].back();
                    for (unsigned val : localResultsRowPointers[t]) {
                        this->_rowOffsets->push_back(val + offset);
                    }
                } else {
                    this->_rowOffsets->insert(this->_rowOffsets->end(), localResultsRowPointers[t].begin(), localResultsRowPointers[t].end());
                }
            }
        }

        void matrixMultiply(NumericalMatrixStorage<T> &inputMatrixData, NumericalMatrixStorage<T>& resultMatrixData, T scaleThis, T scaleOther) override {
            T *thisValues = this->_values->getDataPointer();
            unsigned *thisColumnIndices = this->_columnIndices->getDataPointer();
            unsigned *thisRowPointers = this->_rowOffsets->getDataPointer();

            T *inputValues = inputMatrixData[0];
            unsigned *inputColumnIndices = reinterpret_cast<unsigned*>(inputMatrixData[1]);
            unsigned *inputRowPointers = reinterpret_cast<unsigned*>(inputMatrixData[2]);

            auto matrixMultiplyJob = [&](unsigned startRow, unsigned endRow) {
                for (unsigned row = startRow; row < endRow; ++row) {
                    for (unsigned col = 0; col < this->_numberOfColumns; ++col) {
                        T sum = 0;
                        unsigned thisRowStart = thisRowPointers[row];
                        unsigned thisRowEnd = thisRowPointers[row + 1];
                        for (unsigned k = thisRowStart; k < thisRowEnd; ++k) {
                            unsigned thisCol = thisColumnIndices[k];
                            unsigned inputRowStart = inputRowPointers[thisCol];
                            unsigned inputRowEnd = inputRowPointers[thisCol + 1];
                            for (unsigned l = inputRowStart; l < inputRowEnd; ++l) {
                                if (inputColumnIndices[l] == col) {
                                    sum += scaleThis * thisValues[k] * scaleOther * inputValues[l];
                                    break;
                                }
                            }
                        }
                        resultMatrixData[row * this->_numberOfColumns + col] = sum;
                    }
                }
            };

            this->_threading.executeParallelJob(matrixMultiplyJob, this->_numberOfRows);
        }*/





    private:
            shared_ptr<NumericalVector<unsigned>> _columnIndices;
            shared_ptr<NumericalVector<unsigned>> _rowOffsets;

    };

} // LinearAlgebra

#endif //UNTITLED_COMPRESSEDSPARSEROWMATRIXSTORAGE_H
