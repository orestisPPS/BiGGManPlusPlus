//
// Created by hal9000 on 10/15/23.
//

#ifndef UNTITLED_STEADYSTATEPDEPROPERTIES_H
#define UNTITLED_STEADYSTATEPDEPROPERTIES_H
#include "SecondOrderLinearPDEProperties.h"

namespace MathematicalEntities {
    /**
     * @class SteadyStatePDEProperties
     * @brief Class that encapsulates the properties of a steady state PDE.
     * 
     * This class provides functionalities to set and retrieve spatial properties for 
     * scalar and vector fields. It also allows for both isotropic and anisotropic spatial 
     * settings.
     */
    class SpatialPDEProperties : public SecondOrderLinearPDEProperties {
    public:

        /**
         * @brief Constructor to initialize the class.
         * @param physicalSpaceDimensions Number of physical dimensions.
         * @param fieldType Type of field - Scalar or Vector.
         */
        SpatialPDEProperties(unsigned short physicalSpaceDimensions, FieldType fieldType);

        /**
         * @brief Set isotropic spatial properties for the PDE. The medium has the same properties for
         * all nodes and for every direction.
         * 
         * This function is used to set the isotropic spatial properties for scalar and vector fields.
         * 
         * @param secondOrderCoefficient Coefficient for the second order spatial derivative term.
         * @param firstOrderCoefficient Coefficient for the first order spatial derivative term.
         * @param zerothOrderCoefficient Coefficient for the zeroth order spatial derivative term.
         * @param sourceTerm The source term for the PDE.
         */
        void setIsotropicSpatialProperties(double secondOrderCoefficient, double firstOrderCoefficient,
                                           double zerothOrderCoefficient, double sourceTerm);

        /**
         * @brief Set anisotropic spatial properties for scalar fields. The properties can vary from node to node and
         * from direction to direction.
         * 
         * @param scalarFieldProperties Anisotropic properties for the scalar field.
         * @param nodeId Pointer to the node ID.
         * @throws invalid_argument If the node ID is not found.
         */
        void setAnisotropicSpatialProperties(SpatialScalarFieldPDEProperties scalarFieldProperties, unsigned* nodeId);

        /**
         * @brief Set anisotropic spatial properties for vector fields. The properties can vary from node to node and
         * from direction to direction.
         * 
         * @param vectorFieldProperties Anisotropic properties for the vector field.
         * @param nodeId Pointer to the node ID.
         * @throws invalid_argument If the node ID is not found.
         */
        void setAnisotropicSpatialProperties(SpatialVectorFieldPDEProperties vectorFieldProperties, unsigned* nodeId);

        /**
         * @brief Set locally anisotropic spatial properties for scalar fields. The properties can vary from node to node and
         * from direction to direction.
         * 
         * @param spatialProperties Map of node IDs to their anisotropic properties for scalar fields.
         * @throws invalid_argument If trying to set for a vector field.
         */
        void setLocallyAnisotropicSpatialProperties(shared_ptr<map<unsigned*, SpatialScalarFieldPDEProperties>> spatialProperties);

        /**
         * @brief Set locally anisotropic spatial properties for vector fields. The properties can vary from node to node and
         * from direction to direction.
         * 
         * @param spatialProperties Map of node IDs to their anisotropic properties for vector fields.
         * @throws invalid_argument If trying to set for a scalar field.
         */
        void setLocallyAnisotropicSpatialProperties(shared_ptr<map<unsigned*, SpatialVectorFieldPDEProperties>> spatialProperties);

        /**
         * @brief Retrieve the coefficient of the dependent variable term without node ID.
         * 
         * @param derivativeOrder Order of the derivative (0, 1, or 2).
         * @param direction1 Primary direction (defaults to None).
         * @param direction2 Secondary direction (defaults to None and set equal to direction1).
         * @return Coefficient value.
         * @throws invalid_argument If locally anisotropic properties are set or if derivative order is invalid.
         */
        double getDependentVariableTermCoefficient(unsigned derivativeOrder, Direction direction1 = None, Direction direction2 = None);

        /**
         * @brief Retrieve the coefficient of the dependent variable term with node ID.
         * 
         * @param derivativeOrder Order of the derivative (0, 1, or 2).
         * @param nodeId Pointer to the node ID.
         * @param direction1 Primary direction (defaults to None).
         * @param direction2 Secondary direction (defaults to None).
         * @return Coefficient value.
         * @throws invalid_argument If derivative order is invalid or if required directions are not provided.
         */
        double getDependentVariableTermCoefficient(unsigned derivativeOrder, unsigned* nodeId,
                                                   Direction direction1 = None, Direction direction2 = None);

        /**
         * @brief Retrieve the coefficient of the independent variable term.
         * 
         * @param nodeId Pointer to the node ID.
         * @return Coefficient value.
         */
        double getIndependentVariableTermCoefficient(unsigned* nodeId);
        
    protected:
        
        SpatialVectorFieldPDEProperties _vectorFieldGlobalSpatialProperties;
        
        SpatialScalarFieldPDEProperties _scalarFieldGlobalSpatialProperties;
        
        shared_ptr<map<unsigned*, SpatialScalarFieldPDEProperties>> _locallyAnisotropicScalarFieldSpatialProperties;
        
        shared_ptr<map<unsigned*, SpatialVectorFieldPDEProperties>> _locallyAnisotropicVectorFieldSpatialProperties;
        
    };

} // SecondOrderLinearPDEProperties

#endif //UNTITLED_SPATIALPDEPROPERTIES_H
