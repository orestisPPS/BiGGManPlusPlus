//
// Created by hal9000 on 12/6/22.
//
#pragma once
#include "../Primitives/Array.h"
#include "../Discretization/Node/Node.h"
#include "iostream"
#include "vector"
#include "map"
using namespace std;
using namespace LinearAlgebra;

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
        SecondOrderLinearPDEProperties(Array<double> *secondOrderCoefficients,
                                       vector<double> *firstOrderCoefficients,
                                       double *zerothOrderCoefficient,
                                       double *sourceTerm, bool *isTransient);
            
        SecondOrderLinearPDEProperties(map<int*, Array<double>> *secondOrderCoefficients,
                                       map<int*, vector<double>> *firstOrderCoefficients,
                                       map<int*, double> *zerothOrderCoefficients,
                                       map<int*, double> *sourceTerms, bool *isTransient);
        
        PropertiesDistributionType Type();
        
        bool IsTransient();

        template<class T>
        T *SecondOrderCoefficients();
        
        template<class T>
        T *FirstOrderCoefficients();
        
        template<class T>
        T *ZerothOrderCoefficient();
        
        template<class T>
        T *SourceTerm();

        
    
    private:
        PropertiesDistributionType _type;
        bool *_isTransient;
        
        Array<double> *_secondDerivativeIsotropicProperties;
        vector<double> *_firstDerivativeIsotropicProperties;
        double *_zeroDerivativeIsotropicProperties;
        double *_sourceProperties;
                
        map<int*, Array<double>> *_secondDerivativeLocallyAnisotropicProperties;
        map<int*, vector<double>> *_firstDerivativeLocallyAnisotropicProperties;
        map<int*, double> *_zeroDerivativeLocallyAnisotropicProperties;
        map<int*, double> *_sourceLocallyAnisotropicProperties;
    };

} // PartialDifferentialEquations
