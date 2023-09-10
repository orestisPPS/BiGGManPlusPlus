//
// Created by hal9000 on 9/10/23.
//

#ifndef UNTITLED_NUMERICALMATRIXENUMS_H
#define UNTITLED_NUMERICALMATRIXENUMS_H

namespace LinearAlgebra{
    
    enum MatrixElementsOrder{
        RowMajor,
        ColumnMajor,
        SparseMethodSpecific
    };

    enum NumericalMatrixStorageType {
        FullMatrix,
        CoordinateList,
        CSR,
    };

    enum NumericalMatrixFormType{
        General,
        Symmetric,
        UpperTriangular,
        LowerTriangular,
    };
    
}
#endif //UNTITLED_NUMERICALMATRIXENUMS_H
