//
// Created by hal9000 on 9/3/23.
//

#ifndef UNTITLED_SPARSEMATRIXDATASTORAGEPROVIDER_H
#define UNTITLED_SPARSEMATRIXDATASTORAGEPROVIDER_H

#include "NumericalMatrixStorageDataProvider.h"
#include "NumericalMatrixStorageDataBuilder.h"

namespace LinearAlgebra {

    template<typename T>
    class SparseMatrixDataStorageProvider : public NumericalMatrixStorageDataProvider<T> {
    public:
        SparseMatrixDataStorageProvider(unsigned numberOfRows, unsigned numberOfColumns, NumericalMatrixFormType formType, unsigned availableThreads) :
                NumericalMatrixStorageDataProvider<T>(numberOfRows, numberOfColumns, formType, availableThreads),
                _builder(numberOfRows, numberOfColumns),
                _zero(static_cast<T>(0)) {
            //this->_storageType = NumericalMatrixStorageType::CoordinateList;
            //this->_values = make_shared<NumericalVector<T>>(numberOfRows * numberOfColumns, 0, availableThreads);
        }

    protected:
        NumericalMatrixStorageDataBuilder<T> _builder;
        
        T _zero;
    };

} // LinearAlgebra

#endif //UNTITLED_SPARSEMATRIXDATASTORAGEPROVIDER_H
