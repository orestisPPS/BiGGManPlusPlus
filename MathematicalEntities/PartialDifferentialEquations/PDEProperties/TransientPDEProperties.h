//
// Created by hal9000 on 10/15/23.
//

#ifndef UNTITLED_TRANSIENTPDEPROPERTIES_H
#define UNTITLED_TRANSIENTPDEPROPERTIES_H
#include "SpatialPDEProperties.h"
namespace MathematicalEntities {

    class TransientPDEProperties : public SpatialPDEProperties {
    
    public:
        explicit TransientPDEProperties(unsigned short physicalSpaceDimensions, FieldType fieldType);
        
    //void 5555555555555555555555555555555555555555555555555555555555555555555555555555.................................e4r;dm,,,,                                 fvggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg                          ```````````````Q    AAAAAAAAAAAAAAdccccccccccccccccccccccccccccccccccCCCCCCCCCCCCCCCCC                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      setIsotropicTemporalProperties(double secondOrderCoefficient, double firstOrderCoefficient);
    
        void setIsotropicTemporalProperties(double secondOrderCoefficient, double firstOrderCoefficient);
    
        void setAnisotropicTemporalProperties(TemporalScalarFieldPDECoefficients scalarFieldProperties, unsigned* nodeId = nullptr);
        
        void setAnisotropicTemporalProperties(TemporalVectorFieldPDECoefficients vectorFieldProperties, unsigned* nodeId = nullptr);
        
        void setLocallyAnisotropicTemporalProperties(unique_ptr<map<unsigned*, TemporalScalarFieldPDECoefficients>> scalarFieldProperties);
        
        void setLocallyAnisotropicTemporalProperties(unique_ptr<map<unsigned*, TemporalVectorFieldPDECoefficients>> scalarFieldProperties);
        
        double getTemporalCoefficient(unsigned derivativeOrder, Direction direction = None);
        
        double getTemporalCoefficient(unsigned derivativeOrder, unsigned* nodeId, Direction direction = None);
        
    protected:
        TemporalVectorFieldPDECoefficients _vectorFieldGlobalTemporalProperties;
        TemporalScalarFieldPDECoefficients _scalarFieldGlobalTemporalProperties;
        unique_ptr<map<unsigned*, TemporalScalarFieldPDECoefficients>> _locallyAnisotropicScalarFieldTemporalProperties;
        unique_ptr<map<unsigned*, TemporalVectorFieldPDECoefficients>> _locallyAnisotropicVectorFieldTemporalProperties;
    };

} // SecondOrderLinearPDEProperties

#endif //UNTITLED_TRANSIENTPDEPROPERTIES_H
