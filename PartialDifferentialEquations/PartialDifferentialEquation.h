//
// Created by hal9000 on 12/2/22.
//

#ifndef UNTITLED_DIFFERENTIALEQUATION_H
#define UNTITLED_DIFFERENTIALEQUATION_H

#include "SecondOrderLinearPDEProperties.h"
using namespace PartialDifferentialEquations;

namespace PartialDifferentialEquations {

    enum PDEType{
        GeneralizedSecondOrderLinear,
        EnergyTransfer,
        MassTransfer,
        MomentumTransfer,
        ConvectionDiffusionReaction,
        Laplace,
        Poisson,
        Wave
    };
    
    class PartialDifferentialEquation {
    public:
        PartialDifferentialEquation();
        ~PartialDifferentialEquation();
        SecondOrderLinearPDEProperties *properties;
        PDEType Type();
    };

} // PartialDifferentialEquations

#endif //UNTITLED_DIFFERENTIALEQUATION_H
