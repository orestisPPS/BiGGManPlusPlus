//
// Created by hal9000 on 10/15/23.
//

#ifndef UNTITLED_STEADYSTATEPARTIALDIFFERENTIALEQUATION_H
#define UNTITLED_STEADYSTATEPARTIALDIFFERENTIALEQUATION_H
#include "PartialDifferentialEquation.h"
#include "PDEProperties/SpatialPDEProperties.h"
namespace MathematicalEntities {

    class SteadyStatePartialDifferentialEquation : public PartialDifferentialEquation {
    public:
        explicit SteadyStatePartialDifferentialEquation(shared_ptr<SpatialPDEProperties> properties, PDEType type = GeneralizedSecondOrderLinear);
        
        shared_ptr<SpatialPDEProperties> spatialProperties;
    };

} // MathematicalEntities

#endif //UNTITLED_STEADYSTATEPARTIALDIFFERENTIALEQUATION_H
