//
// Created by hal9000 on 10/15/23.
//

#include "TransientPartialDifferentialEquation.h"

namespace MathematicalEntities {
    TransientPartialDifferentialEquation::TransientPartialDifferentialEquation(shared_ptr<SpatialPDEProperties> spatialProperties,
                                                                               shared_ptr<TransientPDEProperties> transientProperties,
                                                                               PDEType type)
            : SteadyStatePartialDifferentialEquation(std::move(spatialProperties), type), transientProperties(std::move(transientProperties)){
        _type = type;
    }
    
} // MathematicalEntities