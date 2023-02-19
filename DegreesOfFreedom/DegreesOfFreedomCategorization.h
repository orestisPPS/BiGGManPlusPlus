//
// Created by hal9000 on 2/19/23.
//

#ifndef UNTITLED_DEGREESOFFREEDOMCATEGORIZATION_H
#define UNTITLED_DEGREESOFFREEDOMCATEGORIZATION_H

#include "DegreeOfFreedom.h"

namespace DegreeOfFreedom {
    struct DisplacementField1D {
        DegreesOfFreedom::DOFType Displacement1;
    };

    struct DisplacementField2D {
        DegreesOfFreedom::DOFType Displacement1;
    };

    struct DisplacementField3D {
        DegreesOfFreedom::DOFType Displacement1;
        DegreesOfFreedom::DOFType Displacement2;
        DegreesOfFreedom::DOFType Displacement3;
    };
    
    struct TemperatureField {
        DegreesOfFreedom::DOFType Temperature;
    };
    
    struct PressureField1D {
        DegreesOfFreedom::DOFType Pressure1;
    };
    
    struct PressureField2D {
        DegreesOfFreedom::DOFType Pressure1;
        DegreesOfFreedom::DOFType Pressure2;
    };

    struct PressureField3D {
        DegreesOfFreedom::DOFType Pressure1;
        DegreesOfFreedom::DOFType Pressure2;
        DegreesOfFreedom::DOFType Pressure3;
    };

} // DegreeOfFreedom

#endif //UNTITLED_DEGREESOFFREEDOMCATEGORIZATION_H
