//
// Created by hal9000 on 5/5/23.
//

#ifndef UNTITLED_SPACEFIELDPROPERTIES_H
#define UNTITLED_SPACEFIELDPROPERTIES_H
#include "../LinearAlgebra/Array/Array.h"
#include "SecondOrderLinearPDEProperties.h"

namespace PartialDifferentialEquations {
    
/*    enum SpaceFieldProperties {
        Scalar,
        Vector,
        Tensor
    };*/

    struct SpaceFieldProperties {
        shared_ptr<Array<double>> secondOrderCoefficients;
        shared_ptr<vector<double>> firstOrderCoefficients;
        shared_ptr<double> zerothOrderCoefficient;
        shared_ptr<double> sourceTerm;
    };

    struct TimeFieldProperties {
        shared_ptr<Array<double>> secondOrderCoefficients;
        shared_ptr<vector<double>> firstOrderCoefficients;
        shared_ptr<double> zerothOrderCoefficient;
        shared_ptr<double> sourceTerm;
    };
    
} // PartialDifferentialEquations

#endif //UNTITLED_SPACEFIELDPROPERTIES_H
