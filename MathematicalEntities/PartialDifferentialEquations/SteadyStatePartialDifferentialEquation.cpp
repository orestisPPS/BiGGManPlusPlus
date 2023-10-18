//
// Created by hal9000 on 10/15/23.
//

#include "SteadyStatePartialDifferentialEquation.h"

namespace MathematicalEntities {
    SteadyStatePartialDifferentialEquation::SteadyStatePartialDifferentialEquation(
            shared_ptr<SpatialPDEProperties> properties, PDEType type)
            : PartialDifferentialEquation(properties, type), spatialProperties(std::move(properties)){
        _type = type;
    }
} // MathematicalEntities