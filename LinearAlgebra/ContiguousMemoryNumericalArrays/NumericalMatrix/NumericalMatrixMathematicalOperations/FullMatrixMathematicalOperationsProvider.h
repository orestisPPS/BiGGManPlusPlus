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
            
            unsigned &numRows = this->_numberOfRows;
            unsigned &numCols = this->_numberOfColumns;
            unsigned &commonDim = this->_numberOfColumns;
            
            auto multiplyJob = [&](unsigned startRow, unsigned endRow) -> void {
                for (unsigned row = startRow; row < endRow && row < numRows; ++row) {
                    T sum = 0;
                    for (unsigned k = 0; k < commonDim; ++k) {
                        sum += scaleThis * thisValues[row * commonDim + k] * scaleOther * vector[k];
                    }
                    resultVector[row] = sum;
                }
            };
            ThreadingOperations<T>::executeParallelJob(multiplyJob, numRows, availableThreads);
        }

        T vectorMultiplicationRowWisePartial(T *vector, unsigned targetRow, unsigned startColumn, unsigned endColumn,
                                                     T scaleThis, T scaleInput, unsigned availableThreads) override {
            T* thisValues = this->_storageData->getValues()->getDataPointer();
            const unsigned &numCols = this->_numberOfColumns;

            auto size = endColumn - startColumn + 1;
            auto multiplyJob = [&](unsigned start, unsigned end) -> T {
                T sum = 0;
                for (unsigned column = 0; column < size; ++column) {
                    sum += scaleThis * thisValues[targetRow * numCols + (startColumn + column)] * scaleInput * vector[column];
                }
                return sum;
            };
            return ThreadingOperations<T>::executeParallelJobWithReduction(multiplyJob, size, availableThreads);
        }

/*        T vectorMultiplicationRowWisePartial(T *vector, unsigned targetRow, unsigned startColumn, unsigned endColumn,
                                                     bool operationCondition(unsigned i, unsigned j),
                                                     T scaleThis, T scaleOther, unsigned availableThreads) override{
            eThreads);
        }*/

        T vectorMultiplicationColumnWisePartial(T *vector, unsigned targetColumn, unsigned startRow, unsigned endRow,
                                                T scaleThis, T scaleOther, unsigned availableThreads) override {
            T* thisValues = this->_storageData->getValues()->getDataPointer();
            unsigned &numRows = this->_numberOfRows;
            unsigned &numCols = this->_numberOfColumns;

            // Ensure endRow is within bounds of numRows
            endRow = std::min(endRow, numRows);

            auto multiplyJob = [&](unsigned start, unsigned end) -> T {
                T sum = 0;
                for (unsigned row = 0; row < endRow - startRow + 1; ++row) {
                    sum += scaleThis * thisValues[(startRow + row) * numCols + targetColumn] * scaleOther * vector[row];
                }
                return sum;
            };
            return ThreadingOperations<T>::executeParallelJobWithReduction(multiplyJob, endRow - startRow, availableThreads);
        }
        
        

/*        T vectorMultiplicationColumnWisePartial(T *vector, unsigned targetRow, unsigned startColumn, unsigned endColumn,
                                                        bool operationCondition(unsigned i, unsigned j),
                                                        T scaleThis, T scaleOther, unsigned availableThreads) override {
            T* thisValues = this->_storageData->getValues()->getDataPointer();
            unsigned &numCols = this->_numberOfColumns;
            // Ensure endColumn is within bounds
            endColumn = std::min(endColumn, numCols);
                                                              
            auto multiplyJob = [&](unsigned start, unsigned end) -> T {
                T sum = 0;
                for (unsigned column = startColumn; column < end && column < endColumn; ++column) {
                    if (operationCondition(targetRow, column)) {
                        sum += scaleThis * thisValues[targetRow * numCols + column] * scaleOther * vector[column];
                    }
                return sum;
            };
            return ThreadingOperations<T>::executeParallelJobWithReduction(multiplyJob, endColumn - startColumn, availableThreads);
        }*/
        
                                                        
        
    };

} // LinearAlgebra

#endif //UNTITLED_FULLMATRIXMATHEMATICALOPERATIONSPROVIDER_H
