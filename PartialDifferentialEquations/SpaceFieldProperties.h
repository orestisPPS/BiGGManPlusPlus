//
// Created by hal9000 on 5/5/23.
//

#ifndef UNTITLED_SPACEFIELDPROPERTIES_H
#define UNTITLED_SPACEFIELDPROPERTIES_H
#include "../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "SecondOrderLinearPDEProperties.h"

namespace PartialDifferentialEquations {
    
/*    enum SpaceFieldProperties {
        Scalar,
        Vector,
        Tensor
    };*/

    struct SpaceFieldProperties {
        shared_ptr<LinearAlgebra::NumericalMatrix<double>> secondOrderCoefficients;
        shared_ptr<NumericalVector<double>> firstOrderCoefficients;
        shared_ptr<double> zerothOrderCoefficient;
        shared_ptr<double> sourceTerm;
    };

    struct TimeFieldProperties {
        shared_ptr<LinearAlgebra::NumericalMatrix<double>> secondOrderCoefficients;
        shared_ptr<NumericalVector<double>> firstOrderCoefficients;
        shared_ptr<double> zerothOrderCoefficient;
        shared_ptr<double> sourceTerm;
    };
    
} // PartialDifferentialEquations

#endif //UNTITLED_SPACEFIELDPROPERTIES_H
