//
// Created by hal9000 on 12/6/22.
//
#include "SecondOrderLinearPDEProperties.h"
#include "../LinearAlgebra/Array/Array.h"
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
        _localProperties = nullptr;
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
            for (unsigned short i = 0; i < totalDimensions; i++){
                _firstDerivativeProperties->at(i) = firstOrderCoefficient;
                for (int j = 0; j < totalDimensions; ++j) {
                    _secondDerivativeProperties->at(i,j) = secondOrderCoefficient;
                }
            }
            *_zeroDerivativeProperties = zerothOrderCoefficient;
            *_sourceTerm = sourceTerm;
            _isInitialized = true;
        } else {
            //throw argument
            throw std::invalid_argument("Cannot set isotropic properties to a non-isotropic PDE");
        }
    }
    
    void SecondOrderLinearPDEProperties::setFieldAnisotropicProperties(Array<double> *secondOrderCoefficients,
                                                                       vector<double> *firstOrderCoefficients,
                                                                       double zerothOrderCoefficient, double sourceTerm) {
        if (_type == FieldAnisotropic and not _isInitialized){
            _firstDerivativeProperties = firstOrderCoefficients;
            _secondDerivativeProperties = secondOrderCoefficients;
            *_zeroDerivativeProperties = zerothOrderCoefficient;
            *_sourceTerm = sourceTerm;
            _isInitialized = true;
        } else {
            //throw argument
            throw std::invalid_argument("Cannot set field anisotropic properties to a non-field anisotropic PDE");
        }
    }
    
    void SecondOrderLinearPDEProperties::setLocallyAnisotropicProperties(map<unsigned, Array<double>*> *secondOrderCoefficients,
                                                                         map<unsigned, vector<double>*> *firstOrderCoefficients,
                                                                         map<unsigned, double*> *zerothOrderCoefficients,
                                                                         map<unsigned, double*> *sourceTerms) {
        if (_type == LocallyAnisotropic &&
            secondOrderCoefficients->size() == firstOrderCoefficients->size() &&
            firstOrderCoefficients->size() == zerothOrderCoefficients->size() &&
            zerothOrderCoefficients->size() == sourceTerms->size() && 
            !_isInitialized){
            
            _localProperties = new map<unsigned, localProperties>();
            for (auto const& node : *secondOrderCoefficients){
                auto nodeId = node.first;
                _localProperties->insert(pair<unsigned, localProperties>(nodeId, localProperties()));
                _localProperties->at(nodeId).secondOrderCoefficients = node.second;
                _localProperties->at(nodeId).firstOrderCoefficients = firstOrderCoefficients->at(nodeId);
                _localProperties->at(nodeId).zerothOrderCoefficient = zerothOrderCoefficients->at(nodeId);
                _localProperties->at(nodeId).sourceTerm = sourceTerms->at(nodeId);
            }
            _isInitialized = true;
        } else {
            //throw argument
            throw std::invalid_argument("Cannot set locally anisotropic properties to a non-locally anisotropic PDE");
        }
    }

    SecondOrderLinearPDEProperties::localProperties SecondOrderLinearPDEProperties::getLocalProperties(unsigned nodeId) {
        if (_type == LocallyAnisotropic){
            return _localProperties->at(nodeId);
        } else {
            throw std::invalid_argument("Cannot get local properties from a non-locally anisotropic PDE");
        }
    }

    SecondOrderLinearPDEProperties::localProperties SecondOrderLinearPDEProperties::getLocalProperties() {
        if (_type == LocallyAnisotropic){
            throw std::invalid_argument("The PDE is locally anisotropic. Node ID must be specified");
        } else {
            auto localProperties = SecondOrderLinearPDEProperties::localProperties();
            localProperties.secondOrderCoefficients = _secondDerivativeProperties;
            localProperties.firstOrderCoefficients = _firstDerivativeProperties;
            localProperties.zerothOrderCoefficient = _zeroDerivativeProperties;
            localProperties.sourceTerm = _sourceTerm;
            return localProperties;
        }
    }
} // PartialDifferentialEquations