//
// Created by hal9000 on 12/2/22.
//
#pragma once
#ifndef UNTITLED_DIFFERENTIALEQUATION_H
#define UNTITLED_DIFFERENTIALEQUATION_H

#include "PDEProperties/SecondOrderLinearPDEProperties.h"
using namespace MathematicalEntities;

namespace MathematicalEntities {

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
        explicit PartialDifferentialEquation(shared_ptr<SecondOrderLinearPDEProperties> properties, PDEType type);
        shared_ptr<SecondOrderLinearPDEProperties> properties;
        PDEType type();


    protected:
        PDEType _type;
    };

} // MathematicalEntities

#endif //UNTITLED_DIFFERENTIALEQUATION_H
