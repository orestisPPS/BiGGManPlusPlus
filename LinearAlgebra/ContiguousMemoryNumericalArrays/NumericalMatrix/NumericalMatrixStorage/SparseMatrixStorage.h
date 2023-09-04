//
// Created by hal9000 on 9/3/23.
//

#ifndef UNTITLED_SPARSEMATRIXSTORAGE_H
#define UNTITLED_SPARSEMATRIXSTORAGE_H

#include "NumericalMatrixStorage.h"
#include "NumericalMatrixDataStructuresProvider.h"

namespace LinearAlgebra {

    template<typename T>
    class SparseMatrixStorage : public NumericalMatrixStorage<T> {
    public:
        SparseMatrixStorage(NumericalMatrixStorageType storageType, unsigned numberOfRows, unsigned numberOfColumns,
                            ParallelizationMethod parallelizationMethod) :
        NumericalMatrixStorage<T>(storageType, numberOfRows, numberOfColumns, parallelizationMethod){
            _builder = NumericalMatrixDataStructuresProvider<T>(numberOfRows, numberOfColumns);
            _zero = static_cast<T>(0);
            
        }
    protected:
        NumericalMatrixDataStructuresProvider<T> _builder;
        
        T _zero;
    };

} // LinearAlgebra

#endif //UNTITLED_SPARSEMATRIXSTORAGE_H
