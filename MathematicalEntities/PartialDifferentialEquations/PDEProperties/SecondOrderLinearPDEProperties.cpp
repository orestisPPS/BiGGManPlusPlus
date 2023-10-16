//
// Created by hal9000 on 12/6/22.
//
#include "SecondOrderLinearPDEProperties.h"

#include <utility>
#include "../../../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "iostream"
#include "vector"
#include "map"
using namespace std;
using namespace LinearAlgebra;

namespace MathematicalEntities {
    
    SecondOrderLinearPDEProperties::SecondOrderLinearPDEProperties(unsigned short dimensions, FieldType fieldType) :
        _dimensions(dimensions), _fieldType(fieldType) {
        
    }
    
    FieldType SecondOrderLinearPDEProperties::getFieldType() const {
        return _fieldType;
    }
       
    
    
} // MathematicalEntities