//
// Created by hal9000 on 12/2/22.
//
#pragma once
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
        PartialDifferentialEquation(shared_ptr<SecondOrderLinearPDEProperties> properties, PDEType type);
        shared_ptr<SecondOrderLinearPDEProperties> properties;
        PDEType Type();
        
    private:
        PDEType _type;
    };

} // PartialDifferentialEquations

#endif //UNTITLED_DIFFERENTIALEQUATION_H
