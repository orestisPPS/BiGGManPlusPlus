//
// Created by hal9000 on 12/2/22.
//

#include "PartialDifferentialEquation.h"

namespace PartialDifferentialEquations {
    
        PartialDifferentialEquation::PartialDifferentialEquation(SecondOrderLinearPDEProperties* properties, PDEType type) :
        properties(properties), _type(type) { }
        
        PartialDifferentialEquation::~PartialDifferentialEquation() {
            delete properties;
        }
        
        PDEType PartialDifferentialEquation::Type() {
            return _type;
        }
} // PartialDifferentialEquations
