//
// Created by hal9000 on 12/6/22.
//
#include "SecondOrderLinearPDEProperties.h"
#include "../LinearAlgebra/Array.h"
#include "iostream"
#include "vector"
#include "map"
using namespace std;
using namespace LinearAlgebra;

namespace PartialDifferentialEquations {
    
    SecondOrderLinearPDEProperties::SecondOrderLinearPDEProperties(Array<double> *secondOrderCoefficients,
                                                                   vector<double> *firstOrderCoefficients,
                                                                   double *zerothOrderCoefficient,
                                                                   double *sourceTerm, bool *isTransient) {
/*        if (!secondOrderCoefficients->isSymmetric())
        {
            throw std::invalid_argument("Second order coefficients matrix must be symmetric.");
        }
        if (secondOrderCoefficients->numberOfColumns() >4 || secondOrderCoefficients->numberOfRows() > 4) {
            throw std::invalid_argument("Second order derivative coefficients matrix size should not exceed 4x4 (one for each  direction + time)");
        }
        if (firstOrderCoefficients->size() > 4) {
            throw std::invalid_argument("First order derivative coefficients vector size should not exceed 4 (one for each  direction + time)");
        }*/
        //auto lol = secondOrderCoefficients->numberOfColumns();
        _type = PropertiesDistributionType::Isotropic;
        _isTransient = isTransient;
                
        _secondDerivativeIsotropicProperties = secondOrderCoefficients;
        _firstDerivativeIsotropicProperties = firstOrderCoefficients;
        _zeroDerivativeIsotropicProperties = zerothOrderCoefficient;
        _sourceProperties = sourceTerm;
        
        _secondDerivativeLocallyAnisotropicProperties = nullptr;
        _firstDerivativeLocallyAnisotropicProperties = nullptr;
        _zeroDerivativeLocallyAnisotropicProperties = nullptr;
        _sourceLocallyAnisotropicProperties = nullptr;
    }
    
    SecondOrderLinearPDEProperties::SecondOrderLinearPDEProperties(
            map<int *, Array<double>> *secondOrderCoefficients,
            map<int *, vector<double>> *firstOrderCoefficients,
            map<int *, double> *zerothOrderCoefficients,
            map<int *, double> *sourceTerms, bool * isTransient)
    {
        _type = PropertiesDistributionType::LocallyAnisotropic;
        _isTransient = isTransient;
        _secondDerivativeLocallyAnisotropicProperties = secondOrderCoefficients;
        _firstDerivativeLocallyAnisotropicProperties = firstOrderCoefficients;
        _zeroDerivativeLocallyAnisotropicProperties = zerothOrderCoefficients;
        _sourceLocallyAnisotropicProperties = sourceTerms;

        _secondDerivativeIsotropicProperties = nullptr;
        _firstDerivativeIsotropicProperties = nullptr;
        _zeroDerivativeIsotropicProperties = nullptr;
        _sourceProperties = nullptr;
    }

    PropertiesDistributionType SecondOrderLinearPDEProperties::Type() {
        return _type;
    }

    template<class T>
    T *SecondOrderLinearPDEProperties::SecondOrderCoefficients() {
        switch (_type) {
            case PropertiesDistributionType::LocallyAnisotropic:
                return _secondDerivativeLocallyAnisotropicProperties;
            default:
                return _secondDerivativeIsotropicProperties;
        }
    }

    template<class T>
    T *SecondOrderLinearPDEProperties::FirstOrderCoefficients() {
        switch (_type) {
            case PropertiesDistributionType::LocallyAnisotropic:
                return _firstDerivativeLocallyAnisotropicProperties;
            default:
                return _firstDerivativeIsotropicProperties;
        }
    }

    template<class T>
    T *SecondOrderLinearPDEProperties::ZerothOrderCoefficient() {
        switch (_type) {
            case PropertiesDistributionType::LocallyAnisotropic:
                return _zeroDerivativeLocallyAnisotropicProperties;
            default:
                return _zeroDerivativeIsotropicProperties;
        }
    }

    template<class T>
    T *SecondOrderLinearPDEProperties::SourceTerm() {
        switch (_type) {
            case PropertiesDistributionType::LocallyAnisotropic:
                return _sourceLocallyAnisotropicProperties;
            default:
                return _sourceProperties;
        }
    }
} // PartialDifferentialEquations