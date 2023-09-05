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
                
        virtual void matrixAddition(shared_ptr<NumericalMatrixStorageDataProvider<T>>& otherMatrix,
                                    shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                                    T scale1, T scale2) { }
                                    
                                    
        
        virtual void matrixSubtraction(shared_ptr<NumericalMatrixStorageDataProvider<T>>& otherMatrix,
                                       shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                                       T scale1, T scale2) { }
                                       
        virtual void MatrixMultiplication(shared_ptr<NumericalMatrixStorageDataProvider<T>>& otherMatrix,
                                        shared_ptr<NumericalMatrixStorageDataProvider<T>>& resultMatrix,
                                        T scale1, T scale2) { }
                                       
        
                
    protected:
        shared_ptr<NumericalMatrixStorageDataProvider<T>> &_storageData;
        
        unsigned _numberOfRows;
        
        unsigned _numberOfColumns;
        
        T _zero;
        

    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALMATRIXMATHEMATICALOPERATIONSPROVIDER_H
