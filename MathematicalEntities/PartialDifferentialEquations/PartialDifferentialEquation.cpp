//
// Created by hal9000 on 12/2/22.
//

#include "PartialDifferentialEquation.h"

#include <utility>

namespace MathematicalEntities {
    
        PartialDifferentialEquation::PartialDifferentialEquation(shared_ptr<SecondOrderLinearPDEProperties> properties, PDEType type) :
        properties(std::move(properties)) { }
        

    PartialDifferentialEquation::PartialDifferentialEquation(FieldType fieldType, unsigned short dimensions, bool isTransient) :
        _fieldType(fieldType), _dimensions(dimensions), _isTransient(isTransient),
        _spatialDerivativesCoefficients(make_shared<SpatialPDEProperties>(dimensions, fieldType)),
        _temporalDerivativesCoefficients(make_shared<TransientPDEProperties>(dimensions, fieldType)) {
    }

    const FieldType &PartialDifferentialEquation::fieldType() const {
        return _fieldType;
    }

    unsigned short PartialDifferentialEquation::dimensions() const {
        return _dimensions;
    }

    bool PartialDifferentialEquation::isTransient() const {
        return _isTransient;
    }
    
    const shared_ptr<SpatialPDEProperties>& PartialDifferentialEquation::spatialDerivativesCoefficients() const {
        return _spatialDerivativesCoefficients;
    }
    
    const shared_ptr<TransientPDEProperties>& PartialDifferentialEquation::temporalDerivativesCoefficients() const {
        if (!_isTransient) {
            throw runtime_error("This Partial Differential Equation is not transient");
        }
        return _temporalDerivativesCoefficients;
    }
} // MathematicalEntities
