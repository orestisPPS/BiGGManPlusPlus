//
// Created by hal9000 on 12/2/22.
//
#ifndef UNTITLED_DIFFERENTIALEQUATION_H
#define UNTITLED_DIFFERENTIALEQUATION_H

#include "PDEProperties/TransientPDEProperties.h"
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
    
    enum DerivativeOrder{
        None,
        Zeroth,
        First,
        Second
    };
    
    class PartialDifferentialEquation {
    public:
        explicit PartialDifferentialEquation(shared_ptr<SecondOrderLinearPDEProperties> properties, PDEType type);
        shared_ptr<SecondOrderLinearPDEProperties> properties;

        explicit PartialDifferentialEquation(FieldType fieldType, unsigned short dimensions, bool isTransient = false);
        

        
        const FieldType &fieldType() const;
        
        unsigned short dimensions() const;
        
        bool isTransient() const;
        
        const shared_ptr<SpatialPDEProperties>& spatialDerivativesCoefficients() const;
        
        const shared_ptr<TransientPDEProperties>& temporalDerivativesCoefficients() const;
        
        
        


    protected:
        FieldType _fieldType;
        
        unsigned short _dimensions;
        
        bool _isTransient;

        shared_ptr<SpatialPDEProperties> _spatialDerivativesCoefficients;

        shared_ptr<TransientPDEProperties> _temporalDerivativesCoefficients;
        
    };

} // MathematicalEntities

#endif //UNTITLED_DIFFERENTIALEQUATION_H
