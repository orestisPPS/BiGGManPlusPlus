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
                            T scaleThis, T scaleOther) override {

            T* thisValues = this->_storageData->getValues()->getDataPointer();
            T* otherValues = inputMatrix->getValues()->getDataPointer();
            T* resultValues = resultMatrix->getValues()->getDataPointer();
            
            unsigned size = this->_numberOfRows * this->_numberOfColumns;
            
            auto addJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < size; ++i) {
                    resultValues[i] = scaleThis * thisValues[i] + scaleOther * otherValues[i];
                }
            };
            ThreadingOperations<T>::executeParallelJob(addJob, size, this->_storageData->getAvailableThreads());
        }
        
        void matrixSubtraction(shared_ptr<NumericalMatrixStorageDataProvider<T>>& inputMatrix,
                               shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                               T scaleThis, T scaleOther) override {
            
            T* thisValues = this->_storageData->getValues()->getDataPointer();
            T* otherValues = inputMatrix->getValues()->getDataPointer();
            T* resultValues = resultMatrix->getValues()->getDataPointer();
            
            unsigned size = this->_numberOfRows * this->_numberOfColumns;
            
            auto subtractJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end && i < size; ++i) {
                    resultValues[i] = scaleThis * thisValues[i] - scaleOther * otherValues[i];
                }
            };
            ThreadingOperations<T>::executeParallelJob(subtractJob, size, this->_storageData->getAvailableThreads());

        }
        
        void  matrixMultiplication(shared_ptr<NumericalMatrixStorageDataProvider<T>>& inputMatrix,
                                  shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                                  T scaleThis, T scaleOther) override {
            

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
            ThreadingOperations<T>::executeParallelJob(multiplyJob, numRows, this->_storageData->getAvailableThreads());
        }
        
        void matrixVectorMultiplication(T *vector, T *resultVector, T scaleThis, T scaleOther) override {
            
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
            ThreadingOperations<T>::executeParallelJob(multiplyJob, numRows, this->_storageData->getAvailableThreads());
        }
        
        
        
    };

} // LinearAlgebra

#endif //UNTITLED_FULLMATRIXMATHEMATICALOPERATIONSPROVIDER_H
