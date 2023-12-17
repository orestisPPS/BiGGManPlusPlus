//
// Created by hal9000 on 9/6/23.
//

#ifndef UNTITLED_CSRMATHEMATICALOPERATIONSPROVIDER_H
#define UNTITLED_CSRMATHEMATICALOPERATIONSPROVIDER_H
#include "../MatrixStorageDataProviders/CSRStorageDataProvider.h"

namespace LinearAlgebra {

    template<typename T>
    class CSRMathematicalOperationsProvider : public NumericalMatrixMathematicalOperationsProvider<T> {
    public:
        explicit CSRMathematicalOperationsProvider(unsigned numberOfRows, unsigned numberOfColumns,
                                                          shared_ptr<NumericalMatrixStorageDataProvider<T>>& storageData) :
                NumericalMatrixMathematicalOperationsProvider<T>(numberOfRows, numberOfColumns, storageData) {
        }

        void matrixAddition(shared_ptr<NumericalMatrixStorageDataProvider<T>>& inputMatrix,
                            shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                            T scaleThis, T scaleOther, unsigned availableThreads) override {

            unsigned numRows = this->_numberOfRows;

            T *thisValues = this->_storageData->getValues()->getDataPointer();
            unsigned *thisRowPtr = this->_storageData->getSupplementaryVectors()[0]->getDataPointer();
            unsigned *thisColInd = this->_storageData->getSupplementaryVectors()[1]->getDataPointer();

            T *otherValues = inputMatrix->getValues()->getDataPointer();
            unsigned *otherRowPtr = inputMatrix->getSupplementaryVectors()[0]->getDataPointer();
            unsigned *otherColInd = inputMatrix->getSupplementaryVectors()[1]->getDataPointer();

            list<T> resultValues = list<T>();
            auto resultRowPtr = make_shared<NumericalVector<unsigned>>(numRows + 1);
            list<T> resultColInd = list<T>();

            for (unsigned i = 0; i < numRows; ++i) {
                unsigned thisRowStart = thisRowPtr[i];
                unsigned thisRowEnd = thisRowPtr[i + 1];
                unsigned otherRowStart = otherRowPtr[i];
                unsigned otherRowEnd = otherRowPtr[i + 1];

                unsigned thisRowInd = thisRowStart;
                unsigned otherRowInd = otherRowStart;

                while (thisRowInd < thisRowEnd and otherRowInd < otherRowEnd) {
                    unsigned thisCol = thisColInd[thisRowInd];
                    unsigned otherCol = otherColInd[otherRowInd];

                    if (thisCol < otherCol) {
                        resultValues.push_back(scaleThis * thisValues[thisRowInd]);
                        resultColInd.push_back(thisCol);
                        ++thisRowInd;
                    } else if (thisCol > otherCol) {
                        resultValues.push_back(scaleOther * otherValues[otherRowInd]);
                        resultColInd.push_back(otherCol);
                        ++otherRowInd;
                    } else {
                        resultValues.push_back(
                                scaleThis * thisValues[thisRowInd] + scaleOther * otherValues[otherRowInd]);
                        resultColInd.push_back(thisCol);
                        ++thisRowInd;
                        ++otherRowInd;
                    }
                }

                while (thisRowInd < thisRowEnd) {
                    resultValues.push_back(scaleThis * thisValues[thisRowInd]);
                    resultColInd.push_back(thisColInd[thisRowInd]);
                    ++thisRowInd;
                }

                while (otherRowInd < otherRowEnd) {
                    resultValues.push_back(scaleOther * otherValues[otherRowInd]);
                    resultColInd.push_back(otherColInd[otherRowInd]);
                    ++otherRowInd;
                }

                (*resultRowPtr)[i + 1] = resultValues.size();
            }
            
            
            shared_ptr<NumericalVector<T>> resultValuesVector = make_shared<NumericalVector<T>>(resultValues.size());
            shared_ptr<NumericalVector<unsigned>> resultColIndVector = make_shared<NumericalVector<unsigned>>(
                    resultColInd.size());

            unsigned i = 0;
            for (auto value: resultValues) {
                (*resultValuesVector)[i] = value;
                ++i;
            }
            i = 0;
            for (auto value: resultColInd) {
                (*resultColIndVector)[i] = value;
                ++i;
            }
            resultMatrix->setValues(std::move(resultValuesVector));
            resultMatrix->setSupplementaryVectors({std::move(resultRowPtr),std::move(resultColIndVector)});    
        }

        void matrixSubtraction(shared_ptr<NumericalMatrixStorageDataProvider<T>>& inputMatrix,
                            shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                            T scaleThis, T scaleOther, unsigned availableThreads) override {

            unsigned numRows = this->_numberOfRows;

            T *thisValues = this->_storageData->getValues()->getDataPointer();
            unsigned *thisRowPtr = this->_storageData->getSupplementaryVectors()[0]->getDataPointer();
            unsigned *thisColInd = this->_storageData->getSupplementaryVectors()[1]->getDataPointer();

            T *otherValues = inputMatrix->getValues()->getDataPointer();
            unsigned *otherRowPtr = inputMatrix->getSupplementaryVectors()[0]->getDataPointer();
            unsigned *otherColInd = inputMatrix->getSupplementaryVectors()[1]->getDataPointer();

            list<T> resultValues = list<T>();
            auto resultRowPtr = make_shared<NumericalVector<unsigned>>(numRows + 1);
            list<T> resultColInd = list<T>();

            for (unsigned i = 0; i < numRows; ++i) {
                unsigned thisRowStart = thisRowPtr[i];
                unsigned thisRowEnd = thisRowPtr[i + 1];
                unsigned otherRowStart = otherRowPtr[i];
                unsigned otherRowEnd = otherRowPtr[i + 1];

                unsigned thisRowInd = thisRowStart;
                unsigned otherRowInd = otherRowStart;

                while (thisRowInd < thisRowEnd and otherRowInd < otherRowEnd) {
                    unsigned thisCol = thisColInd[thisRowInd];
                    unsigned otherCol = otherColInd[otherRowInd];

                    if (thisCol < otherCol) {
                        resultValues.push_back(scaleThis * thisValues[thisRowInd]);
                        resultColInd.push_back(thisCol);
                        ++thisRowInd;
                    } else if (thisCol > otherCol) {
                        resultValues.push_back(-scaleOther * otherValues[otherRowInd]);
                        resultColInd.push_back(otherCol);
                        ++otherRowInd;
                    } else {
                        resultValues.push_back(
                                scaleThis * thisValues[thisRowInd] - scaleOther * otherValues[otherRowInd]);
                        resultColInd.push_back(thisCol);
                        ++thisRowInd;
                        ++otherRowInd;
                    }
                }

                while (thisRowInd < thisRowEnd) {
                    resultValues.push_back(scaleThis * thisValues[thisRowInd]);
                    resultColInd.push_back(thisColInd[thisRowInd]);
                    ++thisRowInd;
                }
                
                while (otherRowInd < otherRowEnd) {
                    resultValues.push_back(-scaleOther * otherValues[otherRowInd]);
                    resultColInd.push_back(otherColInd[otherRowInd]);
                    ++otherRowInd;
                }

                (*resultRowPtr)[i + 1] = resultValues.size();
            }


            shared_ptr<NumericalVector<T>> resultValuesVector = make_shared<NumericalVector<T>>(resultValues.size());
            shared_ptr<NumericalVector<unsigned>> resultColIndVector = make_shared<NumericalVector<unsigned>>(
                    resultColInd.size());

            unsigned i = 0;
            for (auto value: resultValues) {
                (*resultValuesVector)[i] = value;
                ++i;
            }
            i = 0;
            for (auto value: resultColInd) {
                (*resultColIndVector)[i] = value;
                ++i;
            }
            resultMatrix->setValues(std::move(resultValuesVector));
            resultMatrix->setSupplementaryVectors({std::move(resultRowPtr),std::move(resultColIndVector)});
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
    };

} // LinearAlgebra

#endif //UNTITLED_CSRMATHEMATICALOPERATIONSPROVIDER_H
