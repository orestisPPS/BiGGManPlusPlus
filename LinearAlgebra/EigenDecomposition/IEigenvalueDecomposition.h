//
// Created by hal9000 on 9/9/23.
//

#ifndef UNTITLED_IEIGENVALUEDECOMPOSITION_H
#define UNTITLED_IEIGENVALUEDECOMPOSITION_H

#include "../ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"

namespace LinearAlgebra {
    template<typename T>
    class IEigenvalueDecomposition {
    public:
        virtual T getMostDominantEigenvalue() = 0;
        
        virtual NumericalVector<T> getMostDominantEigenvector() = 0;
        
        virtual NumericalVector<T> getEigenvalues() = 0;
        
        virtual map<T, NumericalVector<T>> getEigenvalueToEigenvectorMap() = 0;
    };

} // LinearAlgebra

#endif //UNTITLED_IEIGENVALUEDECOMPOSITION_H
