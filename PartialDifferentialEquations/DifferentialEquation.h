//
// Created by hal9000 on 12/2/22.
//

#ifndef UNTITLED_DIFFERENTIALEQUATION_H
#define UNTITLED_DIFFERENTIALEQUATION_H

#include "SecondOrderLinearPDEProperties.h"
using namespace PartialDifferentialEquations;

namespace PartialDifferentialEquations {

    enum PDEType{
        SteadyStateLinearElasticity,
        TransientLinearElasticity,
        SteadyStateHeatTransfer,
        TransientHeatTransfer,
        SteadyStateMassTransfer,
        TransientMassTransfer,
        SteadyStateMomentumTransfer,
        TransientMomentumTransfer,
        SteadyStateConvectionDiffusionReaction,
        TransientConvectionDiffusionReaction,
        Laplace,
        Poisson,
        Wave
    };
    
    class PartialDifferentialEquation {
    public:
        PartialDifferentialEquation();
        ~PartialDifferentialEquation();
    };

} // PartialDifferentialEquations

#endif //UNTITLED_DIFFERENTIALEQUATION_H
