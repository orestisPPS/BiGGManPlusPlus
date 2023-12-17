//
// Created by hal9000 on 9/6/23.
//

#ifndef UNTITLED_NUMERICALMATRIXMATHEMATICALOPERATIONSPROVIDER_H
#define UNTITLED_NUMERICALMATRIXMATHEMATICALOPERATIONSPROVIDER_H

#include "../MatrixStorageDataProviders/NumericalMatrixStorageDataProvider.h"

namespace LinearAlgebra {

    template<typename T>
    class NumericalMatrixMathematicalOperationsProvider {
    public:
        explicit NumericalMatrixMathematicalOperationsProvider(unsigned numberOfRows, unsigned numberOfColumns,
                shared_ptr<NumericalMatrixStorageDataProvider<T>>& storageData) :
                _numberOfRows(numberOfRows), _numberOfColumns(numberOfColumns), _storageData(storageData){ }
                
        virtual void matrixScalarMultiplication(T scaleThis) {
            if (static_cast<T>(0) == scaleThis) {
                //TODO: create zero() method in NumericalMatrixStorageDataProvider
            }
            else
                _storageData->getValues()->scale(scaleThis);
        }
                
        virtual void matrixAddition(shared_ptr<NumericalMatrixStorageDataProvider<T>>& inputMatrix,
                                    shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                                    T scaleThis, T scaleOther, unsigned usedDefinedThreads) { }
                                    
        virtual void matrixSubtraction(shared_ptr<NumericalMatrixStorageDataProvider<T>>& otherMatrix,
                                       shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                                       T scaleThis, T scaleOther, unsigned usedDefinedThreads) { }
                                       
        virtual void matrixMultiplication(shared_ptr<NumericalMatrixStorageDataProvider<T>>& otherMatrix,
                                          shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                                          T scaleThis, T scaleOther, unsigned usedDefinedThreads) { }
                                          
        virtual void vectorMultiplication(T *vector, T *resultVector, T scaleThis, T scaleOther, unsigned usedDefinedThreads) { }

        virtual T vectorMultiplicationRowWisePartial(T *vector, unsigned targetRow, unsigned startColumn, unsigned endColumn,
                                                     T scaleThis, T scaleInput, unsigned availableThreads) { }
        
        virtual T vectorMultiplicationRowWisePartial(T *vector, unsigned targetRow, unsigned startColumn, unsigned endColumn,
                                                     bool operationCondition(unsigned i, unsigned j),
                                                     T scaleThis, T scaleOther, unsigned availableThreads) { }

        virtual T vectorMultiplicationColumnWisePartial(T *vector, unsigned targetRow, unsigned startColumn, unsigned endColumn,
                                                     T scaleThis, T scaleInput, unsigned availableThreads) { }

        virtual T vectorMultiplicationColumnWisePartial(T *vector, unsigned targetRow, unsigned startColumn, unsigned endColumn,
                                                     bool operationCondition(unsigned i, unsigned j),
                                                     T scaleThis, T scaleOther, unsigned availableThreads) { }
        
        
        virtual void Axpy(T *vector, T *resultVector, T scaleThis, T scaleMultipliedVector, T scaleAddedVector) { }
                                        
        virtual void matrixTranspose(shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix) { }
        
        virtual void inverse() { }
        

                                                
                                       
        
                
    protected:
        shared_ptr<NumericalMatrixStorageDataProvider<T>>& _storageData;
        
        unsigned _numberOfRows;
        
        unsigned _numberOfColumns;
        
        T _zero;
        

    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALMATRIXMATHEMATICALOPERATIONSPROVIDER_H
