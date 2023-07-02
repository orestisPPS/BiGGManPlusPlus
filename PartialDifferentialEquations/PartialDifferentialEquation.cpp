//
// Created by hal9000 on 12/2/22.
//

#include "PartialDifferentialEquation.h"

#include <utility>

namespace PartialDifferentialEquations {
    
        PartialDifferentialEquation::PartialDifferentialEquation(shared_ptr<SecondOrderLinearPDEProperties> properties, PDEType type) :
        properties(std::move(properties)), _type(type) { }
        
        PDEType PartialDifferentialEquation::Type() {
            return _type;
        }
} // PartialDifferentialEquations
