//
// Created by hal9000 on 10/15/23.
//

#ifndef UNTITLED_TRANSIENTPDEPROPERTIES_H
#define UNTITLED_TRANSIENTPDEPROPERTIES_H
#include "SecondOrderLinearPDEProperties.h"
namespace MathematicalEntities {

    class TransientPDEProperties : public SecondOrderLinearPDEProperties {
        
        TransientPDEProperties(unsigned short physicalSpaceDimensions, FieldType fieldType);
        
        
    };

} // SecondOrderLinearPDEProperties

#endif //UNTITLED_TRANSIENTPDEPROPERTIES_H
