//
// Created by hal9000 on 10/15/23.
//

#include "SpatialPDEProperties.h"

namespace MathematicalEntities {
    SpatialPDEProperties::SpatialPDEProperties(unsigned short physicalSpaceDimensions, FieldType fieldType) :
        SecondOrderLinearPDEProperties(physicalSpaceDimensions, fieldType) {
        
    }
    
    void SpatialPDEProperties::setIsotropicSpatialProperties(double secondOrderCoefficient, double firstOrderCoefficient,
                                                             double zerothOrderCoefficient, double sourceTerm) {
        if (_fieldType == ScalarField) {

            auto secondOrderCoefficients = make_unique<NumericalMatrix<double>>(_dimensions, _dimensions);
            auto firstOrderCoefficients = make_unique<NumericalVector<double>>(_dimensions);
            for (unsigned short i = 0; i < _dimensions; i++) {
                secondOrderCoefficients->setElement(i, i, secondOrderCoefficient);
                (*firstOrderCoefficients)[i] = firstOrderCoefficient;
            }
            _scalarFieldGlobalSpatialProperties.secondOrderCoefficients = std::move(secondOrderCoefficients);
            _scalarFieldGlobalSpatialProperties.firstOrderCoefficients = std::move(firstOrderCoefficients);
            _scalarFieldGlobalSpatialProperties.zerothOrderCoefficient = make_unique<double>(zerothOrderCoefficient);
            _scalarFieldGlobalSpatialProperties.sourceTerm = make_unique<double>(sourceTerm);
        }
        else{
            auto secondOrderCoefficients = make_unique<NumericalMatrix<double>>(_dimensions, _dimensions);
            auto firstOrderCoefficients = make_unique<NumericalMatrix<double>>(_dimensions, _dimensions);
            auto zerothOrderCoefficients = make_unique<NumericalVector<double>>(_dimensions);
            for (unsigned short i = 0; i < _dimensions; i++) {
                (*zerothOrderCoefficients)[i] = zerothOrderCoefficient;
                for (unsigned short j = 0; j < _dimensions; j++) {
                    secondOrderCoefficients->setElement(i, j, secondOrderCoefficient);
                    firstOrderCoefficients->setElement(i, j, firstOrderCoefficient);
                }
            }
            _vectorFieldGlobalSpatialProperties.secondOrderCoefficients = std::move(secondOrderCoefficients);
            _vectorFieldGlobalSpatialProperties.firstOrderCoefficients = std::move(firstOrderCoefficients);
            _vectorFieldGlobalSpatialProperties.zerothOrderCoefficient = std::move(zerothOrderCoefficients);
            _vectorFieldGlobalSpatialProperties.sourceTerm = make_unique<double>(sourceTerm);
        }
    }
    
    void SpatialPDEProperties::setAnisotropicSpatialProperties(SpatialScalarFieldPDEProperties scalarFieldProperties, unsigned* nodeId) {
        switch (_fieldType) {
            case ScalarField:
                if (_locallyAnisotropicScalarFieldSpatialProperties != nullptr && nodeId != nullptr)
                    _locallyAnisotropicScalarFieldSpatialProperties->at(nodeId) = std::move(scalarFieldProperties);
                else
                    throw invalid_argument("SteadyStatePDEProperties::setAnisotropicSpatialProperties:"
                                           " Node ID not found. If new nodes need to be assigned use setLocallyAnisotropicSpatialProperties instead. ");
                break;
            case VectorField:
                throw invalid_argument(
                        "SteadyStatePDEProperties::setAnisotropicSpatialProperties: Cannot set anisotropic spatial properties for a vector field.");
                break;
        }
    }
    
    void SpatialPDEProperties::setAnisotropicSpatialProperties(SpatialVectorFieldPDEProperties vectorFieldProperties, unsigned* nodeId) {
        switch (_fieldType) {
            case VectorField:
                if (_locallyAnisotropicVectorFieldSpatialProperties != nullptr && nodeId != nullptr)
                    _locallyAnisotropicVectorFieldSpatialProperties->at(nodeId) = std::move(vectorFieldProperties);
                else
                    throw invalid_argument("SteadyStatePDEProperties::setAnisotropicSpatialProperties:"
                                           " Node ID not found. If new nodes need to be assigned use setLocallyAnisotropicSpatialProperties instead. ");
                break;
            case ScalarField:
                throw invalid_argument(
                        "SteadyStatePDEProperties::setAnisotropicSpatialProperties: Cannot set anisotropic spatial properties for a scalar field.");
                break;
        }
    }
    
    void SpatialPDEProperties::setLocallyAnisotropicSpatialProperties(shared_ptr<map<unsigned*, SpatialScalarFieldPDEProperties>> spatialProperties) {
        if (_fieldType == ScalarField) {
            _locallyAnisotropicScalarFieldSpatialProperties = std::move(spatialProperties);
        }
        else
            throw invalid_argument("SteadyStatePDEProperties::setLocallyAnisotropicSpatialProperties: Cannot set anisotropic spatial properties for a vector field.");
    }
    
    
    void SpatialPDEProperties::setLocallyAnisotropicSpatialProperties(shared_ptr<map<unsigned*, SpatialVectorFieldPDEProperties>> spatialProperties) {
        if (_fieldType == VectorField) {
            _locallyAnisotropicVectorFieldSpatialProperties = std::move(spatialProperties);
        }
        else
            throw invalid_argument("SteadyStatePDEProperties::setLocallyAnisotropicSpatialProperties: Cannot set anisotropic spatial properties for a scalar field.");
    }
    
    double SpatialPDEProperties::getDependentVariableTermCoefficient(unsigned derivativeOrder, Direction direction1, Direction direction2) {
        if (_locallyAnisotropicScalarFieldSpatialProperties != nullptr || _locallyAnisotropicVectorFieldSpatialProperties != nullptr)
            throw invalid_argument("SteadyStatePDEProperties::getDependentVariableTermCoefficient: "
                                   "Locally anisotropic spatial properties set. Use other overloaded function with node ID instead.");
        
        if (direction2 == None)
            direction2 = direction1;
        unsigned i = spatialDirectionToUnsigned[direction1];
        unsigned j = spatialDirectionToUnsigned[direction2];
        switch (derivativeOrder) {
            case 2:
                return (_fieldType == ScalarField)
                       ? _scalarFieldGlobalSpatialProperties.secondOrderCoefficients->getElement(i, j)
                       : _vectorFieldGlobalSpatialProperties.secondOrderCoefficients->getElement(i, j);
            case 1:
                return (_fieldType == ScalarField)
                       ? (*_scalarFieldGlobalSpatialProperties.firstOrderCoefficients)[i]
                       : _vectorFieldGlobalSpatialProperties.firstOrderCoefficients->getElement(i, j);
            case 0:
                return (_fieldType == ScalarField)
                       ? *_scalarFieldGlobalSpatialProperties.zerothOrderCoefficient
                       : (*_vectorFieldGlobalSpatialProperties.zerothOrderCoefficient)[i];
            default:
                throw invalid_argument("SteadyStatePDEProperties::getDependentVariableTermCoefficient: "
                                       "Derivative order must be 0, 1 or 2.");
        }

    }
    
    double SpatialPDEProperties::getDependentVariableTermCoefficient(unsigned derivativeOrder, unsigned* nodeId, Direction direction1, Direction direction2) {
        if (direction2 == None)
            direction2 = direction1;
        unsigned i = spatialDirectionToUnsigned[direction1];
        unsigned j = spatialDirectionToUnsigned[direction2];
        auto handleScalarSecondOrder = [&]() {
            if (direction1 == None)
                throw invalid_argument("SteadyStatePDEProperties::getDependentVariableTermCoefficient: "
                                       "Direction 1 must be specified for the second order spatial coefficients"
                                       "of a scalar field.");
            if (_locallyAnisotropicScalarFieldSpatialProperties != nullptr) {
                return _locallyAnisotropicScalarFieldSpatialProperties->at(nodeId).secondOrderCoefficients->getElement(i, j);
            }
            return _scalarFieldGlobalSpatialProperties.secondOrderCoefficients->getElement(i, j);
        };

        auto handleVectorSecondOrder = [&]() {
            if (direction1 == None)
                throw invalid_argument("SteadyStatePDEProperties::getDependentVariableTermCoefficient: "
                                       "Direction 1 must be specified for the second order spatial coefficients"
                                       "of a vector field.");
            if (_locallyAnisotropicVectorFieldSpatialProperties != nullptr) {
                return _locallyAnisotropicVectorFieldSpatialProperties->at(nodeId).secondOrderCoefficients->getElement(i, j);
            }
            return _vectorFieldGlobalSpatialProperties.secondOrderCoefficients->getElement(i, j);
        };

        auto handleScalarFirstOrder = [&]() {
            if (direction1 == None)
                throw invalid_argument("SteadyStatePDEProperties::getDependentVariableTermCoefficient: "
                                       "Direction 1 must be specified for the first order spatial coefficients"
                                       "of a scalar field.");
            if (_locallyAnisotropicScalarFieldSpatialProperties != nullptr) {
                return (*_locallyAnisotropicScalarFieldSpatialProperties->at(nodeId).firstOrderCoefficients)[i];
            }
            return (*_scalarFieldGlobalSpatialProperties.firstOrderCoefficients)[i];
        };

        auto handleVectorFirstOrder = [&]() {
            if (direction1 == None)
                throw invalid_argument("SteadyStatePDEProperties::getDependentVariableTermCoefficient: "
                                       "Direction 1 must be specified for the first order spatial coefficients"
                                       "of a vector field.");
            
            if (_locallyAnisotropicVectorFieldSpatialProperties != nullptr) {
                return _locallyAnisotropicVectorFieldSpatialProperties->at(nodeId).firstOrderCoefficients->getElement(i, j);
            }
            return _vectorFieldGlobalSpatialProperties.firstOrderCoefficients->getElement(i, j);
        };

        auto handleScalarZerothOrder = [&]() {
            if (_locallyAnisotropicScalarFieldSpatialProperties != nullptr) {
                return *_locallyAnisotropicScalarFieldSpatialProperties->at(nodeId).zerothOrderCoefficient;
            }
            return *_scalarFieldGlobalSpatialProperties.zerothOrderCoefficient;
        };

        auto handleVectorZerothOrder = [&]() {
            if (direction1 == None)
                throw invalid_argument("SteadyStatePDEProperties::getDependentVariableTermCoefficient: "
                                       "Direction 1 must be specified for the zeroth order spatial coefficients"
                                       "of a vector field.");
            if (_locallyAnisotropicVectorFieldSpatialProperties != nullptr) {
                return (*_locallyAnisotropicVectorFieldSpatialProperties->at(nodeId).zerothOrderCoefficient)[i];
            }
            return (*_vectorFieldGlobalSpatialProperties.zerothOrderCoefficient)[i];
        };

        switch (derivativeOrder) {
            case 2:
                return (_fieldType == ScalarField) ? handleScalarSecondOrder() : handleVectorSecondOrder();
            case 1:
                return (_fieldType == ScalarField) ? handleScalarFirstOrder() : handleVectorFirstOrder();
            case 0:
                return (_fieldType == ScalarField) ? handleScalarZerothOrder() : handleVectorZerothOrder();
            default:
                throw invalid_argument("SteadyStatePDEProperties::getDependentVariableTermCoefficient: "
                                       "Derivative order must be 0, 1 or 2.");
        }
    }
    
    double SpatialPDEProperties::getIndependentVariableTermCoefficient(unsigned* nodeId) {
        if (nodeId == nullptr)
            return *_scalarFieldGlobalSpatialProperties.sourceTerm;
        if (_locallyAnisotropicScalarFieldSpatialProperties != nullptr)
            return *_locallyAnisotropicScalarFieldSpatialProperties->at(nodeId).sourceTerm;
        if (_locallyAnisotropicVectorFieldSpatialProperties != nullptr) 
            return *_locallyAnisotropicVectorFieldSpatialProperties->at(nodeId).sourceTerm;

    }
    
} // SecondOrderLinearPDEProperties