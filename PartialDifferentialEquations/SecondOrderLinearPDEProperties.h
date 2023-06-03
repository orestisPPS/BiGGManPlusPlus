//
// Created by hal9000 on 12/6/22.
//
#pragma once

#include "../Discretization/Node/Node.h"
#include "iostream"
#include "map"
#include "FieldProperties.h"
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
    
        ~SecondOrderLinearPDEProperties();
        
        PropertiesDistributionType Type();
        
        bool IsTransient() const;
        
        short unsigned totalDimensions() const;
        
        void setIsotropicProperties(double secondOrderCoefficient, double firstOrderCoefficient,
                                    double zerothOrderCoefficient, double sourceTerm);
        
        void setFieldAnisotropicProperties(FieldProperties globalProperties);
        
        void setLocallyAnisotropicProperties(map<unsigned, FieldProperties> *localProperties);
        

        FieldProperties getLocalProperties(unsigned nodeId);
        
        FieldProperties getLocalProperties();
        



    protected:
        
        PropertiesDistributionType _type;
        
        bool _isTransient;
        
        short unsigned _physicalSpaceDimensions;
        
        Array<double> *_secondDerivativeProperties;
        
        vector<double> *_firstDerivativeProperties;
        
        double *_zeroDerivativeProperties;
        
        double *_sourceTerm;
        
        map<unsigned, FieldProperties> *_locallyAnisotropic1Properties;
        
        bool _isInitialized;

    };

} // PartialDifferentialEquations
