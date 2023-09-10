//
// Created by hal9000 on 9/6/23.
//

#ifndef UNTITLED_FULLMATRIXMATHEMATICALOPERATIONSPROVIDER_H
#define UNTITLED_FULLMATRIXMATHEMATICALOPERATIONSPROVIDER_H

 #include "NumericalMatrixMathematicalOperationsProvider.h"
namespace LinearAlgebra {
    
    template<typename T>
    class FullMatrixMathematicalOperationsProvider : public NumericalMatrixMathematicalOperationsProvider<T> {
    public:
        explicit FullMatrixMathematicalOperationsProvider(unsigned numberOfRows, unsigned numberOfColumns,
                shared_ptr<NumericalMatrixStorageDataProvider<T>>& storageData) :
                NumericalMatrixMathematicalOperationsProvider<T>(numberOfRows, numberOfColumns, storageData) {
        }
                
        void matrixAddition(shared_ptr<NumericalMatrixStorageDataProvider<T>>& inputMatrix,
                            shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                            T scaleThis, T scaleOther, unsigned availableThreads) override {
            
            
            T* thisValues = this->_storageData->getValues()->getDataPointer();
            T* otherValues = inputMatrix->getValues()->getDataPointer();
            T* resultValues = resultMatrix->getValues()->getDataPointer();
            
            unsigned size = this->_numberOfRows * this->_numberOfColumns;
            
            auto addJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < size; ++i) {
                    resultValues[i] = scaleThis * thisValues[i] + scaleOther * otherValues[i];
                }
            };
            ThreadingOperations<T>::executeParallelJob(addJob, size, availableThreads);
        }
        
        void matrixSubtraction(shared_ptr<NumericalMatrixStorageDataProvider<T>>& inputMatrix,
                               shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                               T scaleThis, T scaleOther, unsigned availableThreads) override {
            
            T* thisValues = this->_storageData->getValues()->getDataPointer();
            T* otherValues = inputMatrix->getValues()->getDataPointer();
            T* resultValues = resultMatrix->getValues()->getDataPointer();
            
            unsigned size = this->_numberOfRows * this->_numberOfColumns;
            
            auto subtractJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < size; ++i) {
                    resultValues[i] = scaleThis * thisValues[i] - scaleOther * otherValues[i];
                }
            };
            ThreadingOperations<T>::executeParallelJob(subtractJob, size, availableThreads);

        }
        
        void  matrixMultiplication(shared_ptr<NumericalMatrixStorageDataProvider<T>>& inputMatrix,
                                  shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                                  T scaleThis, T scaleOther, unsigned availableThreads) override {

            T* thisValues = this->_storageData->getValues()->getDataPointer();
            T* otherValues = inputMatrix->getValues()->getDataPointer();
            T* resultValues = resultMatrix->getValues()->getDataPointer();

            unsigned numRows = this->_numberOfRows;
            unsigned numCols = this->_numberOfColumns;
            unsigned commonDim = this->_numberOfColumns;

            auto multiplyJob = [&](unsigned startRow, unsigned endRow) -> void {
                for (unsigned i = startRow; i < endRow && i < numRows; ++i) {
                    for (unsigned j = 0; j < numCols; ++j) {
                        T sum = 0;
                        for (unsigned k = 0; k < commonDim; ++k) {
                            sum += scaleThis * thisValues[i * commonDim + k] * scaleOther * otherValues[k * numCols + j];
                        }
                        resultValues[i * numCols + j] = sum;
                    }
                }
            };
            
            ThreadingOperations<T>::executeParallelJob(multiplyJob, numRows, availableThreads);
        }
        
        void vectorMultiplication(T *vector, T *resultVector, T scaleThis, T scaleOther, unsigned availableThreads) override {
            
            T* thisValues = this->_storageData->getValues()->getDataPointer();
            
            unsigned numRows = this->_numberOfRows;
            unsigned numCols = this->_numberOfColumns;
            unsigned commonDim = this->_numberOfColumns;
            
            auto multiplyJob = [&](unsigned startRow, unsigned endRow) -> void {
                for (unsigned i = startRow; i < endRow && i < numRows; ++i) {
                    T sum = 0;
                    for (unsigned k = 0; k < commonDim; ++k) {
                        sum += scaleThis * thisValues[i * commonDim + k] * scaleOther * vector[k];
                    }
                    resultVector[i] = sum;
                }
            };
            ThreadingOperations<T>::executeParallelJob(multiplyJob, numRows, availableThreads);
        }

        T vectorMultiplicationRowWisePartial(T *vector, T scaleThis, T scaleOther,
                                             unsigned targetRow, unsigned startColumn, unsigned endColumn,
                                             bool operationCondition(unsigned i, unsigned j), unsigned availableThreads) override {
            T* thisValues = this->_storageData->getValues()->getDataPointer();
            unsigned numCols = this->_numberOfColumns;
            T sum = 0;
            
            // Ensure endColumn is within bounds
            endColumn = std::min(endColumn, numCols);
            
            auto multiplyJob = [&](unsigned start, unsigned end) -> T {
                for (unsigned j = startColumn; j < end && j < endColumn; ++j) {
                    if (operationCondition(targetRow, j)) {
                        sum += scaleThis * thisValues[targetRow * numCols + j] * scaleOther * vector[j];
                    }
                }
                return sum;
            };
            return ThreadingOperations<T>::executeParallelJobWithReduction(multiplyJob, endColumn - startColumn, availableThreads);
        }

        T vectorMultiplicationColumnWisePartial(T *vector, T scaleThis, T scaleOther,
                                                unsigned targetColumn, unsigned startRow, unsigned endRow,
                                                bool operationCondition(unsigned i, unsigned j), unsigned availableThreads) override {
            T* thisValues = this->_storageData->getValues()->getDataPointer();
            unsigned numRows = this->_numberOfRows;
            unsigned numCols = this->_numberOfColumns;

            // Ensure endRow is within bounds
            endRow = std::min(endRow, numRows);

            auto multiplyJob = [&](unsigned start, unsigned end) -> T {
                T localSum = 0; // Each thread will have its local sum
                for (unsigned i = start; i < end && i < endRow; ++i) {
                    if (operationCondition(i, targetColumn)) {
                        localSum += scaleThis * thisValues[i * numCols + targetColumn] * scaleOther * vector[i];
                    }
                }
                return localSum; // Return the local sum for the reduction operation
            };
            return ThreadingOperations<T>::executeParallelJobWithReduction(multiplyJob, endRow - startRow, availableThreads);
        }


        
        
    };

} // LinearAlgebra

#endif //UNTITLED_FULLMATRIXMATHEMATICALOPERATIONSPROVIDER_H
