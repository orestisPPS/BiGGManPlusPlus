//
// Created by hal9000 on 9/3/23.
//

#ifndef UNTITLED_SPARSEMATRIXSTORAGE_H
#define UNTITLED_SPARSEMATRIXSTORAGE_H

#include "NumericalMatrixStorage.h"
#include "NumericalMatrixDataStructuresBuilder.h"

namespace LinearAlgebra {

    template<typename T>
    class SparseMatrixStorage : public NumericalMatrixStorage<T> {
    public:
        SparseMatrixStorage(unsigned numberOfRows, unsigned numberOfColumns, unsigned availableThreads) :
                NumericalMatrixStorage<T>(numberOfRows, numberOfColumns, availableThreads),
                _builder(numberOfRows, numberOfColumns),
                _zero(static_cast<T>(0)) {
            this->_storageType = NumericalMatrixStorageType::CoordinateList;
            this->_values = make_shared<NumericalVector<T>>(numberOfRows * numberOfColumns, 0, availableThreads);
        }

    protected:
        NumericalMatrixDataStructuresBuilder<T> _builder;
        
        T _zero;
    };

} // LinearAlgebra

#endif //UNTITLED_SPARSEMATRIXSTORAGE_H
