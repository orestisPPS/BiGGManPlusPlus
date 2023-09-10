//
// Created by hal9000 on 9/10/23.
//

#ifndef UNTITLED_NUMERICALMATRIXDATASTORAGEACCESSPROVIDER_H
#define UNTITLED_NUMERICALMATRIXDATASTORAGEACCESSPROVIDER_H
#include "../CSRStorageDataProvider.h"
#include "../FullMatrixStorageDataProvider.h"

namespace LinearAlgebra {

    class NumericalMatrixDataStorageAccessProvider {
        NumericalMatrixDataStorageAccessProvider(NumericalMatrixFormType formType, unsigned numberOfRows, unsigned numberOfColumns,
                                                 unsigned availableThreads) :
                _formType(formType), _numberOfRows(numberOfRows), _numberOfColumns(numberOfColumns),
                _availableThreads(availableThreads) {
            if (formType == General) {
                _storageType = NumericalMatrixStorageType::FullMatrix;
                _values = make_shared<NumericalVector<double>>(numberOfRows * numberOfColumns, 0, availableThreads);
            }
            else if (formType == Symmetric || formType == UpperTriangular || formType == LowerTriangular) {
                _storageType = NumericalMatrixStorageType::CSR;
                _values = make_shared<NumericalVector<double>>(numberOfRows * (numberOfColumns + 1) / 2, 0, availableThreads);
            }
            else {
                throw runtime_error("Form type not recognized.");
            }
        }
    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALMATRIXDATASTORAGEACCESSPROVIDER_H
