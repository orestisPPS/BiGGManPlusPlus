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
        explicit CompressedSparseRowMatrixStorage(unsigned numberOfRows, unsigned numberOfColumns, ParallelizationMethod parallelizationMethod) :
                NumericalMatrixStorage<T>(CSR, numberOfRows, numberOfColumns, parallelizationMethod) {
                this->_values = make_shared<NumericalVector<T>>();
                this->_columnIndices = make_shared<NumericalVector<unsigned>>();
                this->_rowPointers = make_shared<NumericalVector<unsigned>>(numberOfRows + 1);
                (*_rowPointers)[0] = 0;
        }
        
        vector<NumericalVector<T>&> getNecessaryStorageVectors() override {
            return {*this->_values, *this->_columnIndices, *this->_rowPointers};
        }

        void initializeElementAssignment() override {
            if (this->_elementAssignmentRunning)
                throw runtime_error("Element assignment is already running. Call finalizeElementAssignment() first.");

            this->_elementAssignmentRunning = true;
            this->_cooHashMap->clear();
            
            this->_values->getData()->clear();
            _columnIndices->getData()->clear();
            _rowPointers->getData()->clear();
            
            this->_values->resize(0);
            this->_columnIndices->resize(0);
            
            this->_rowPointers->getData()->push_back(0);
        }

        void finalizeElementAssignment() override {
            if (!this->_elementAssignmentRunning) {
                throw runtime_error("Element assignment is not running. Call initializeElementAssignment() first.");
            }
            
            this->_values->getData()->resize(this->_cooHashMap->size());
            this->_columnIndices->getData()->resize(this->_cooHashMap->size());
            
            unsigned i = 0;
            auto thisValuesData = this->_values->getDataPointer();
            auto thisColumnIndicesData = this->_columnIndices->getDataPointer();
            auto thisRowPointersData = this->_rowPointers->getDataPointer();
            
            for (auto& element : *this->_cooHashMap){
                thisValuesData[i] = *element.second;
                thisColumnIndicesData[i] = element.first.second;
                i++;
            }
            
        }

        void setElement(unsigned int row, unsigned int column, const T &value) override {
            if (!this->_elementAssignmentRunning) {
                throw runtime_error("Element assignment is not running. Call initializeElementAssignment() first.");
            }

            unsigned int rowStart = (*_rowPointers)[row];
            unsigned int rowEnd = (*_rowPointers)[row + 1];

            // Find the position where the column should be (or is) in columnIndices
            unsigned int position = rowStart;
            while (position < rowEnd && (*_columnIndices)[position] < column) {
                position++;
            }

            // If we found the column index, just update the value
            if (position < rowEnd && (*_columnIndices)[position] == column)
                (*_values)[position] = value;
            else {
                // Otherwise, we need to insert a new value and column index
                _values->getData()->insert(_values->getData()->begin() + position, value);
                _columnIndices->getData()->insert(_columnIndices->getData()->begin() + position, column);

                // Update the row pointers for this row and all subsequent rows
                for (unsigned int r = row + 1; r <= this->_numberOfRows; r++) {
                    (*_rowPointers)[r]++;
                }
            }
        }

        shared_ptr<NumericalVector<T>> getRowSharedPtr(unsigned row) override {
            if (row >= this->_numberOfRows) {
                throw runtime_error("Row index out of bounds.");
            }
            auto rowVector = make_shared<NumericalVector<T>>(this->_numberOfColumns);
            auto vectorData = rowVector->getDataPointer();
            unsigned int rowStart = (*_rowPointers)[row];
            unsigned int rowEnd = (*_rowPointers)[row + 1];

            for (unsigned int i = 0; i < this->_numberOfColumns; i++) {
                vectorData[i] = 0; // Initialize all elements to zero
            }

            for (unsigned int i = rowStart; i < rowEnd; i++) {
                vectorData[(*_columnIndices)[i]] = (*_values)[i];
            }

            return rowVector;
        }

        shared_ptr<NumericalVector<T>> getColumnSharedPtr(unsigned column) override {
            if (column >= this->_numberOfColumns) {
                throw runtime_error("Column index out of bounds.");
            }
            auto columnVector = make_shared<NumericalVector<T>>(this->_numberOfRows);
            auto vectorData = columnVector->getDataPointer();

            for (unsigned int row = 0; row < this->_numberOfRows; row++) {
                unsigned int rowStart = (*_rowPointers)[row];
                unsigned int rowEnd = (*_rowPointers)[row + 1];
                vectorData[row] = 0; // Initialize to zero
                for (unsigned int i = rowStart; i < rowEnd; i++) {
                    if ((*_columnIndices)[i] == column) {
                        vectorData[row] = (*_values)[i];
                        break;
                    }
                }
            }
            return columnVector;
        }

        void matrixAdd(NumericalMatrixStorage<T> &inputMatrixData,
                       NumericalMatrixStorage<T> &resultMatrixData,
                       T scaleThis, T scaleOther) override {

            vector<T>& thisValues = *this->_values->getData();
            vector<unsigned>& thisColumnIndices = *this->_columnIndices->getData();
            vector<unsigned>& thisRowPointers = *this->_rowPointers->getData();

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
            vector<unsigned>& thisRowPointers = *this->_rowPointers->getData();

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
                        this->_rowPointers->push_back(val + offset);
                    }
                } else {
                    this->_rowPointers->insert(this->_rowPointers->end(), localResultsRowPointers[t].begin(), localResultsRowPointers[t].end());
                }
            }
        }
        
        void matrixSubtract(NumericalMatrixStorage<T> &inputMatrixData,
                            NumericalMatrixStorage<T> &resultMatrixData,
                            T scaleThis, T scaleOther) override {

            vector<T>& thisValues = *this->_values->getData();
            vector<unsigned>& thisColumnIndices = *this->_columnIndices->getData();
            vector<unsigned>& thisRowPointers = *this->_rowPointers->getData();

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
            vector<unsigned>& thisRowPointers = *this->_rowPointers->getData();

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
                        this->_rowPointers->push_back(val + offset);
                    }
                } else {
                    this->_rowPointers->insert(this->_rowPointers->end(), localResultsRowPointers[t].begin(), localResultsRowPointers[t].end());
                }
            }
        }

        void matrixMultiply(NumericalMatrixStorage<T> &inputMatrixData, NumericalMatrixStorage<T>& resultMatrixData, T scaleThis, T scaleOther) override {
            T *thisValues = this->_values->getDataPointer();
            unsigned *thisColumnIndices = this->_columnIndices->getDataPointer();
            unsigned *thisRowPointers = this->_rowPointers->getDataPointer();

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
        }





    private:
            shared_ptr<NumericalVector<unsigned>> _columnIndices;
            shared_ptr<NumericalVector<unsigned>> _rowPointers;

    };

} // LinearAlgebra

#endif //UNTITLED_COMPRESSEDSPARSEROWMATRIXSTORAGE_H
