//
// Created by hal9000 on 10/15/23.
//

#ifndef UNTITLED_STEADYSTATEPDEPROPERTIES_H
#define UNTITLED_STEADYSTATEPDEPROPERTIES_H
#include "SecondOrderLinearPDEProperties.h"

namespace MathematicalEntities {

    class SteadyStatePDEProperties : public SecondOrderLinearPDEProperties {
    public:
        explicit SteadyStatePDEProperties(unsigned short physicalSpaceDimensions, FieldType fieldType = FieldType::ScalarField);
        
        void setIsotropicSpatialProperties(double secondOrderCoefficient, double firstOrderCoefficient,
                                           double zerothOrderCoefficient, double sourceTerm);
        
        void setAnisotropicSpatialProperties(unique_ptr<SpatialScalarFieldPDEProperties> scalarFieldProperties, unsigned* nodeId = nullptr);
        
        void setAnisotropicSpatialProperties(unique_ptr<SpatialVectorFieldPDEProperties> vectorFieldProperties, unsigned* nodeId = nullptr);
        
        void setLocallyAnisotropicSpatialProperties(unique_ptr<map<unsigned*, unique_ptr<SpatialScalarFieldPDEProperties>>> spatialProperties);
        
        void setLocallyAnisotropicSpatialProperties(unique_ptr<map<unsigned*, unique_ptr<SpatialVectorFieldPDEProperties>>> spatialProperties);
        
        double getDependentVariableTermCoefficient(unsigned derivativeOrder, Direction direction1 = None, Direction direction2 = None);
        
        double getDependentVariableTermCoefficient(unsigned derivativeOrder, unsigned* nodeId, Direction direction1 = None, Direction direction2 = None);
        
        double getIndependentVariableTermCoefficient(unsigned* nodeId = nullptr);
        
    protected:
        
        unique_ptr<SpatialVectorFieldPDEProperties> _vectorFieldGlobalSpatialProperties;
        
        unique_ptr<SpatialScalarFieldPDEProperties> _scalarFieldGlobalSpatialProperties;
        
        unique_ptr<map<unsigned*, unique_ptr<SpatialScalarFieldPDEProperties>>> _locallyAnisotropicScalarFieldSpatialProperties;
        
        unique_ptr<map<unsigned*, unique_ptr<SpatialVectorFieldPDEProperties>>> _locallyAnisotropicVectorFieldSpatialProperties;
        
    };

} // SecondOrderLinearPDEProperties

#endif //UNTITLED_STEADYSTATEPDEPROPERTIES_H
