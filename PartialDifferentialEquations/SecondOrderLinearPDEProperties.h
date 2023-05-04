//
// Created by hal9000 on 12/6/22.
//
#pragma once
#include "../LinearAlgebra/Array/Array.h"
#include "../Discretization/Node/Node.h"
#include "iostream"
#include "map"
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
        
        void setFieldAnisotropicProperties(Array<double> *secondOrderCoefficients,
                                           vector<double> *firstOrderCoefficients,
                                           double zerothOrderCoefficient, double sourceTerm);
        
        void setLocallyAnisotropicProperties(map<unsigned, Array<double>*> *secondOrderCoefficients,
                                             map<unsigned, vector<double>*> *firstOrderCoefficients,
                                             map<unsigned, double*> *zerothOrderCoefficients,
                                             map<unsigned, double*> *sourceTerms);
        struct localProperties
        {
            Array<double> *secondOrderCoefficients;
            vector<double> *firstOrderCoefficients;
            double *zerothOrderCoefficient;
            double *sourceTerm;
        };
        localProperties getLocalProperties(unsigned nodeId);
        
        localProperties getLocalProperties();
        



    protected:
        
        PropertiesDistributionType _type;
        
        bool _isTransient;
        
        short unsigned _physicalSpaceDimensions;
        
        Array<double> *_secondDerivativeProperties;
        
        vector<double> *_firstDerivativeProperties;
        
        double *_zeroDerivativeProperties;
        
        double *_sourceTerm;
        
        map<unsigned, localProperties> *_localProperties{};  
        
        bool _isInitialized;

    };

} // PartialDifferentialEquations
