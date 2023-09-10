//
// Created by hal9000 on 12/6/22.
//
#include "SecondOrderLinearPDEProperties.h"

#include <utility>
#include "../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "iostream"
#include "vector"
#include "map"
using namespace std;
using namespace LinearAlgebra;

namespace PartialDifferentialEquations {
    
    SecondOrderLinearPDEProperties::SecondOrderLinearPDEProperties(unsigned short physicalSpaceDimensions, bool isTransient,
                                                                   PropertiesDistributionType type) {
        _physicalSpaceDimensions = physicalSpaceDimensions;
        _isTransient = isTransient;
        _isInitialized = false;
        switch (type) {
            case Isotropic:
                _type = Isotropic;
                break;
            case FieldAnisotropic:
                _type = FieldAnisotropic;
                _sourceTerm = nullptr;
                break;
            case LocallyAnisotropic:
                _type = LocallyAnisotropic;
                break;
        }
        _secondDerivativeProperties = nullptr;
        _firstDerivativeProperties = nullptr;
        _zeroDerivativeProperties = nullptr;
        _sourceTerm = nullptr;
        _locallyAnisotropic1Properties = nullptr;
    }
    
    
    PropertiesDistributionType SecondOrderLinearPDEProperties::Type(){
        return _type;
    }
    
    bool SecondOrderLinearPDEProperties::IsTransient() const{
        return _isTransient;
    }
    
    short unsigned SecondOrderLinearPDEProperties::totalDimensions() const{
        short unsigned totalDimensions = 0;
        if (_isTransient){
            totalDimensions = _physicalSpaceDimensions + 1;
        } else {
            totalDimensions = _physicalSpaceDimensions;
        }
        return totalDimensions;
    }
    
    
    void SecondOrderLinearPDEProperties::setIsotropicProperties(double secondOrderCoefficient, double firstOrderCoefficient,
                                                                double zerothOrderCoefficient, double sourceTerm) {
        if (_type == Isotropic and not _isInitialized){
            auto totalDimensions = this->totalDimensions();
            _firstDerivativeProperties = make_shared<NumericalVector<double> >(totalDimensions);
            _secondDerivativeProperties = make_shared<NumericalMatrix<double> >(totalDimensions, totalDimensions);
            for (unsigned short i = 0; i < totalDimensions; i++){
                _firstDerivativeProperties->at(i) = firstOrderCoefficient;
                for (int j = 0; j < totalDimensions; ++j) {
                    _secondDerivativeProperties->setElement(i, j, secondOrderCoefficient);
                }
            }
            _zeroDerivativeProperties = make_shared<double>(zerothOrderCoefficient);
            _sourceTerm = make_shared<double>(sourceTerm);
            _isInitialized = true;
        } else {
            //throw argument
            throw std::invalid_argument("Cannot set isotropic properties to a non-isotropic PDE");
        }
    }
    
    void SecondOrderLinearPDEProperties::setFieldAnisotropicProperties(SpaceFieldProperties globalProperties) {
        if (_type == FieldAnisotropic and not _isInitialized){
            _firstDerivativeProperties = globalProperties.firstOrderCoefficients;
            _secondDerivativeProperties = globalProperties.secondOrderCoefficients;
            *_zeroDerivativeProperties = *globalProperties.zerothOrderCoefficient; 
            *_sourceTerm = *globalProperties.sourceTerm;
            _isInitialized = true;
        } else {
            //throw argument
            throw std::invalid_argument("Cannot set field anisotropic properties to a non-field anisotropic PDE");
        }
    }
    
    void SecondOrderLinearPDEProperties::setLocallyAnisotropicProperties(
            shared_ptr<map<unsigned int, SpaceFieldProperties>> properties) {
        if (_type == LocallyAnisotropic and not _isInitialized) {
            _locallyAnisotropic1Properties = std::move(properties);
        }
    }

    SpaceFieldProperties SecondOrderLinearPDEProperties::getLocalProperties(unsigned nodeId) {
        if (_type == LocallyAnisotropic){
            return _locallyAnisotropic1Properties->at(nodeId);
        } else {
            auto localProperties = SpaceFieldProperties();
            localProperties.secondOrderCoefficients = _secondDerivativeProperties;
            localProperties.firstOrderCoefficients = _firstDerivativeProperties;
            localProperties.zerothOrderCoefficient = _zeroDerivativeProperties;
            localProperties.sourceTerm = _sourceTerm;
            return localProperties;
        }
    }
    
    SpaceFieldProperties SecondOrderLinearPDEProperties::getLocalProperties() {
        if (_type == LocallyAnisotropic){
            throw std::invalid_argument("The PDE is locally anisotropic. Node ID must be specified");
        } else {
            auto localProperties = SpaceFieldProperties();
            localProperties.secondOrderCoefficients = _secondDerivativeProperties;
            localProperties.firstOrderCoefficients = _firstDerivativeProperties;
            localProperties.zerothOrderCoefficient = _zeroDerivativeProperties;
            localProperties.sourceTerm = _sourceTerm;
            return localProperties;
        }
    }
} // PartialDifferentialEquations