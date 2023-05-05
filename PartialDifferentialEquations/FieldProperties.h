//
// Created by hal9000 on 5/5/23.
//

#ifndef UNTITLED_FIELDPROPERTIES_H
#define UNTITLED_FIELDPROPERTIES_H
#include "../LinearAlgebra/Array/Array.h"
#include "SecondOrderLinearPDEProperties.h"

namespace PartialDifferentialEquations {
    
/*    enum FieldProperties {
        Scalar,
        Vector,
        Tensor
    };*/

    struct FieldProperties {
        Array<double> *secondOrderCoefficients;
        vector<double> *firstOrderCoefficients;
        double *zerothOrderCoefficient;
        double *sourceTerm;
    };
    
} // PartialDifferentialEquations

#endif //UNTITLED_FIELDPROPERTIES_H
