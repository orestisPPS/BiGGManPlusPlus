//
// Created by hal9000 on 10/15/23.
//

#ifndef UNTITLED_TRANSIENTPARTIALDIFFERENTIALEQUATION_H
#define UNTITLED_TRANSIENTPARTIALDIFFERENTIALEQUATION_H
#include "SteadyStatePartialDifferentialEquation.h"
#include "PDEProperties/TransientPDEProperties.h"
namespace MathematicalEntities {

    class TransientPartialDifferentialEquation : public SteadyStatePartialDifferentialEquation {
    public:
        explicit TransientPartialDifferentialEquation(shared_ptr<SpatialPDEProperties> spatialProperties,
                                                      shared_ptr<TransientPDEProperties> transientProperties,
                                                      PDEType type = GeneralizedSecondOrderLinear);
        
        shared_ptr<TransientPDEProperties> transientProperties;
    };

} // MathematicalEntities

#endif //UNTITLED_TRANSIENTPARTIALDIFFERENTIALEQUATION_H
