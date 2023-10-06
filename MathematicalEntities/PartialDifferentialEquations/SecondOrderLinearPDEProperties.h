//
// Created by hal9000 on 12/6/22.
//
#pragma once

#include "../../Discretization/Node/Node.h"
#include "iostream"
#include "map"
#include "SpaceFieldProperties.h"
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
    class SecondOrderLinearPDEProperties {
        
    public :
        SecondOrderLinearPDEProperties(unsigned short physicalSpaceDimensions);
         
        PropertiesDistributionType SpatialPropertiesDistributionType();

        PropertiesDistributionType TimePropertiesDistributionType();
        
        bool IsTransient() const;
        
        short unsigned spatialDimensions() const;
        
        void setIsotropicSpatialProperties(double secondOrderCoefficient, double firstOrderCoefficient,
                                           double zerothOrderCoefficient, double sourceTerm);
        
        void setFieldAnisotropicSpatialProperties(SpaceFieldProperties globalProperties);
        
        void setLocallyAnisotropicSpatialProperties(shared_ptr<map<unsigned int, SpaceFieldProperties>> properties);
        
        void setIsotropicTimeProperties(double secondOrderCoefficient, double firstOrderCoefficient);
        
        void setFieldAnisotropicTimeProperties(TimeFieldProperties globalProperties);
        
        void setLocallyAnisotropicTimeProperties(shared_ptr<map<unsigned int, TimeFieldProperties>> properties);

        SpaceFieldProperties getLocalSpatialProperties (unsigned nodeId = -1);
        
        TimeFieldProperties getLocalTimeProperties (unsigned nodeId = -1);
        



    protected:
        
        PropertiesDistributionType _spatialPropertiesType;
        
        PropertiesDistributionType _timePropertiesType;
        
        bool _isTransient;
        
        short unsigned _dimensions;
        
        shared_ptr<NumericalMatrix<double>> _secondSpatialDerivativeProperties;
        
        shared_ptr<NumericalVector<double>> _firstSpatialDerivativeProperties;

        shared_ptr<NumericalMatrix<double>> _secondTimeDerivativeProperties;

        shared_ptr<NumericalVector<double>> _firstTimeDerivativeProperties;
        
        shared_ptr<double> _zeroDerivativeProperties;
        
        shared_ptr<double> _sourceTerm;
        
        shared_ptr<map<unsigned, SpaceFieldProperties>> _locallyAnisotropicSpatialProperties;
        
        shared_ptr<map<unsigned, TimeFieldProperties>> _locallyAnisotropicTimeProperties;
        
        bool _isInitialized;

    };

} // MathematicalEntities
