//
// Created by hal9000 on 12/6/22.
//
#include "SecondOrderLinearPDEProperties.h"

#include <utility>
#include "../../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "iostream"
#include "vector"
#include "map"
using namespace std;
using namespace LinearAlgebra;

namespace MathematicalEntities {
    
    SecondOrderLinearPDEProperties::SecondOrderLinearPDEProperties(unsigned short dimensions) {
        _dimensions = dimensions;
        _isInitialized = false;
        _firstSpatialDerivativeProperties = nullptr;
        _secondSpatialDerivativeProperties = nullptr;
        _firstTimeDerivativeProperties = nullptr;
        _secondTimeDerivativeProperties = nullptr;
        _zeroDerivativeProperties = nullptr;
        _sourceTerm = nullptr;
        _locallyAnisotropicSpatialProperties = nullptr;
        _locallyAnisotropicTimeProperties = nullptr;
        _isTransient = false;
    }
    
    PropertiesDistributionType SecondOrderLinearPDEProperties::SpatialPropertiesDistributionType(){
        return _spatialPropertiesType;
    }
    
    PropertiesDistributionType SecondOrderLinearPDEProperties::TimePropertiesDistributionType(){
        return _timePropertiesType;
    }
    
    bool SecondOrderLinearPDEProperties::IsTransient() const{
        return _isTransient;
    }
    
    short unsigned SecondOrderLinearPDEProperties::spatialDimensions() const{
        return _dimensions;
    }
    
    void SecondOrderLinearPDEProperties::setIsotropicSpatialProperties(double secondOrderCoefficient, double firstOrderCoefficient,
                                                                       double zerothOrderCoefficient, double sourceTerm) {
            _spatialPropertiesType = Isotropic;
            _firstSpatialDerivativeProperties = make_shared<NumericalVector<double> >(_dimensions);
            _secondSpatialDerivativeProperties = make_shared<NumericalMatrix<double> >(_dimensions, _dimensions);
            for (unsigned short i = 0; i < _dimensions; i++){
                _firstSpatialDerivativeProperties->at(i) = firstOrderCoefficient;
                for (int j = 0; j < _dimensions; ++j) {
                    _secondSpatialDerivativeProperties->setElement(i, j, secondOrderCoefficient);
                }
            }
            _zeroDerivativeProperties = make_shared<double>(zerothOrderCoefficient);
            _sourceTerm = make_shared<double>(sourceTerm);
            _isInitialized = true;
    }
    
    void SecondOrderLinearPDEProperties::setFieldAnisotropicSpatialProperties(SpaceFieldProperties globalProperties) {
        _spatialPropertiesType = FieldAnisotropic;
        _firstSpatialDerivativeProperties = std::move(globalProperties.firstOrderCoefficients);
        _secondSpatialDerivativeProperties = std::move(globalProperties.secondOrderCoefficients);
        _zeroDerivativeProperties = std::move(globalProperties.zerothOrderCoefficient);
        _sourceTerm = std::move(globalProperties.sourceTerm);
        _isInitialized = true;
    }
    
    void SecondOrderLinearPDEProperties::setLocallyAnisotropicSpatialProperties(shared_ptr<map<unsigned int, SpaceFieldProperties>> properties) {
        _spatialPropertiesType = LocallyAnisotropic;
        _locallyAnisotropicSpatialProperties = std::move(properties);
    }
    
    void SecondOrderLinearPDEProperties::setIsotropicTimeProperties(double secondOrderCoefficient, double firstOrderCoefficient) {
        _timePropertiesType = Isotropic;
        _isTransient = true;
        _firstTimeDerivativeProperties = make_shared<NumericalVector<double> >(_dimensions);
        _secondTimeDerivativeProperties = make_shared<NumericalMatrix<double> >(_dimensions, _dimensions);
        for (unsigned short i = 0; i < _dimensions; i++){
            _firstTimeDerivativeProperties->at(i) = firstOrderCoefficient;
            for (int j = 0; j < _dimensions; ++j) {
                _secondTimeDerivativeProperties->setElement(i, j, secondOrderCoefficient);
            }
        }
        _isInitialized = true;
    }
    
    void SecondOrderLinearPDEProperties::setFieldAnisotropicTimeProperties(TimeFieldProperties globalProperties) {
        _timePropertiesType = FieldAnisotropic;
        _isTransient = true;
        _firstTimeDerivativeProperties = std::move(globalProperties.firstOrderCoefficients);
        _secondTimeDerivativeProperties = std::move(globalProperties.secondOrderCoefficients);
        _isInitialized = true;
    }
    
    void SecondOrderLinearPDEProperties::setLocallyAnisotropicTimeProperties(shared_ptr<map<unsigned int,
                                                                             TimeFieldProperties>> properties) {
        _timePropertiesType = LocallyAnisotropic;
        _isTransient = true;
        _locallyAnisotropicTimeProperties = std::move(properties);
    }

    SpaceFieldProperties SecondOrderLinearPDEProperties::getLocalSpatialProperties(unsigned nodeId) {
        if (_spatialPropertiesType == LocallyAnisotropic && nodeId != -1){
            return _locallyAnisotropicSpatialProperties->at(nodeId);
        } else {
            auto localProperties = SpaceFieldProperties();
            localProperties.secondOrderCoefficients = _secondSpatialDerivativeProperties;
            localProperties.firstOrderCoefficients = _firstSpatialDerivativeProperties;
            localProperties.zerothOrderCoefficient = _zeroDerivativeProperties;
            localProperties.sourceTerm = _sourceTerm;
            return localProperties;
        }
    }
    
    TimeFieldProperties SecondOrderLinearPDEProperties::getLocalTimeProperties(unsigned int nodeId) {
        if (_timePropertiesType == LocallyAnisotropic && nodeId != -1){
            return _locallyAnisotropicTimeProperties->at(nodeId);
        } else {
            auto localProperties = TimeFieldProperties();
            localProperties.secondOrderCoefficients = _secondTimeDerivativeProperties;
            localProperties.firstOrderCoefficients = _firstTimeDerivativeProperties;
            return localProperties;
        }
    }
} // MathematicalEntities