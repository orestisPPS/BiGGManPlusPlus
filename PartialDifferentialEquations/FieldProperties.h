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
        shared_ptr<Array<double>> secondOrderCoefficients;
        shared_ptr<vector<double>> firstOrderCoefficients;
        shared_ptr<double> zerothOrderCoefficient;
        shared_ptr<double> sourceTerm;
    };
    
} // PartialDifferentialEquations

#endif //UNTITLED_FIELDPROPERTIES_H
