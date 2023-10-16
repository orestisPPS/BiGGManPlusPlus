//
// Created by hal9000 on 12/6/22.
//
#pragma once

#include "../../../Discretization/Node/Node.h"
#include "iostream"
#include "map"
#include "../SpatialPDEProperties.h"
using namespace std;
using namespace LinearAlgebra;
using namespace Discretization;

using namespace Discretization;

namespace MathematicalEntities {
    
    enum PropertiesDistributionType
    {
        Isotropic,
        FieldAnisotropic,
        LocallyAnisotropic
    };

    enum FieldType{
        ScalarField,
        VectorField,
    };

    class SecondOrderLinearPDEProperties {
        
    public :
        explicit SecondOrderLinearPDEProperties(unsigned short physicalSpaceDimensions, FieldType fieldType);
    
        FieldType getFieldType() const;
    protected:
        short unsigned _dimensions;
        
        FieldType _fieldType;

    };

} // MathematicalEntities
