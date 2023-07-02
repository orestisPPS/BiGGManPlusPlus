//
// Created by hal9000 on 12/6/22.
//
#pragma once

#include "../Discretization/Node/Node.h"
#include "iostream"
#include "map"
#include "SpaceFieldProperties.h"
using namespace std;
using namespace LinearAlgebra;
using namespace Discretization;

using namespace Discretization;

namespace PartialDifferentialEquations {
    enum PropertiesDistributionType
    {
        Isotropic,
        FieldAnisotropic,
        LocallyAnisotropic
    };
    class SecondOrderLinearPDEProperties {
        
    public :
        SecondOrderLinearPDEProperties(unsigned short physicalSpaceDimensions, bool isTransient,
                                       PropertiesDistributionType type);
         
        PropertiesDistributionType Type();
        
        bool IsTransient() const;
        
        short unsigned totalDimensions() const;
        
        void setIsotropicProperties(double secondOrderCoefficient, double firstOrderCoefficient,
                                    double zerothOrderCoefficient, double sourceTerm);
        
        void setFieldAnisotropicProperties(SpaceFieldProperties globalProperties);
        
        void setLocallyAnisotropicProperties(shared_ptr<map<unsigned int, SpaceFieldProperties>> properties);
        

        SpaceFieldProperties getLocalProperties(unsigned nodeId);
        
        SpaceFieldProperties getLocalProperties();
        



    protected:
        
        PropertiesDistributionType _type;
        
        bool _isTransient;
        
        short unsigned _physicalSpaceDimensions;
        
        shared_ptr<Array<double>> _secondDerivativeProperties;
        
        shared_ptr<vector<double>> _firstDerivativeProperties;
        
        shared_ptr<double> _zeroDerivativeProperties;
        
        shared_ptr<double> _sourceTerm;
        
        shared_ptr<map<unsigned, SpaceFieldProperties>> _locallyAnisotropic1Properties;
        
        bool _isInitialized;

    };

} // PartialDifferentialEquations
