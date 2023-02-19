//
// Created by hal9000 on 11/28/22.
//

#ifndef UNTITLED_DEGREEOFFREEDOM_H
#define UNTITLED_DEGREEOFFREEDOM_H

#include <tuple>
#include <iostream>
#include "limits"
#include "DegreeOfFreedomID.h"
using namespace std;
using namespace DegreesOfFreedom;


namespace DegreesOfFreedom{
    enum class DOFType {
        Position1,
        Position2,
        Position3,
        Temperature,
        Pressure1,
        Pressure2,
        Pressure3,
        Displacement1,
        Displacement2,
        Displacement3,
        Rotation1,
        Rotation2,
        Rotation3,
        Velocity1,
        Velocity2,
        Velocity3,
        UnknownScalarVariable,
        UnknownVectorFieldVariableComponent1,
        UnknownVectorFieldVariableComponent2,
        UnknownVectorFieldVariableComponent3,
    };
    
    enum class DisplacementVectorField1D_DOFType {
        Displacement1 = static_cast<int>(DOFType::Displacement1)
    };
    
    enum class DisplacementVectorField2D_DOFType {
        Displacement1 = static_cast<int>(DOFType::Displacement1),
        Displacement2 = static_cast<int>(DOFType::Displacement2),
        Rotation3 = static_cast<int>(DOFType::Rotation3),
    };
        
    enum class DisplacementFieldVector3D_DOFType {
        Displacement1 = static_cast<int>(DOFType::Displacement1),
        Displacement2 = static_cast<int>(DOFType::Displacement2),
        Displacement3 = static_cast<int>(DOFType::Displacement3),
        Rotation1 = static_cast<int>(DOFType::Rotation1),
        Rotation2 = static_cast<int>(DOFType::Rotation2),
        Rotation3 = static_cast<int>(DOFType::Rotation3),
    };
    
    enum class TemperatureScalarField_DOFType {
        Temperature = static_cast<int>(DOFType::Temperature)
    };
    
    enum class VelocityVectorField1D_DOFType {
        Velocity1 = static_cast<int>(DOFType::Velocity1),
    };
    
    enum class VelocityVectorField2D_DOFType {
        Velocity1 = static_cast<int>(DOFType::Velocity1),
        Velocity2 = static_cast<int>(DOFType::Velocity2),
    };
    
    enum class VelocityVectorField3D_DOFType {
        Velocity1 = static_cast<int>(DOFType::Velocity1),
        Velocity2 = static_cast<int>(DOFType::Velocity2),
        Velocity3 = static_cast<int>(DOFType::Velocity3),
    };
    
    enum class PressureVectorField1D_DOFType {
        Pressure1 = static_cast<int>(DOFType::Pressure1),
    };
    
    enum class PressureVectorField2D_DOFType {
        Pressure1 = static_cast<int>(DOFType::Pressure1),
        Pressure2 = static_cast<int>(DOFType::Pressure2),
    };
    
    enum class PressureVectorField3D_DOFType {
        Pressure1 = static_cast<int>(DOFType::Pressure1),
        Pressure2 = static_cast<int>(DOFType::Pressure2),
        Pressure3 = static_cast<int>(DOFType::Pressure3),
    };
    
    enum class UnknownScalarField_DOFType {
        UnknownScalarVariable = static_cast<int>(DOFType::UnknownScalarVariable),
    };
    
    enum class UnknownVectorField1D_DOFType {
        UnknownVectorFieldVariableComponent1 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent1),
    };
    
    enum class UnknownVectorField2D_DOFType {
        UnknownVectorFieldVariableComponent1 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent1),
        UnknownVectorFieldVariableComponent2 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent2),
    };
    
    enum class UnknownVectorField3D_DOFType {
        UnknownVectorFieldVariableComponent1 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent1),
        UnknownVectorFieldVariableComponent2 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent2),
        UnknownVectorFieldVariableComponent3 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent3),
    };
    
    class DegreeOfFreedom {
        public:
            //Use this constructor when the degree of freedom is not fixed. DOFs with constraint type free or flux
            // will be initialized with a value of NaN.
            DegreeOfFreedom(DOFType dofType, DegreeOfFreedomID *id, unsigned* parentNodeGlobalId);
    
            //Use this constructor when the degree of freedom is fixed. Only DOFs with constraint type fixed
            // can be initialized with this constructor.
            DegreeOfFreedom(DOFType dofType, double value, DegreeOfFreedomID *id, unsigned* parentNodeGlobalId);
    
            ~DegreeOfFreedom();
    
            bool operator == (const DegreeOfFreedom& dof);
    
            bool operator != (const DegreeOfFreedom& dof);
    
            //Pointer to the id of the degree of freedom. Contains ConstraintType Enum (fixed, flux, free)
            // and unsigned int id. The enumeration of the id corresponds to the constraint type.
            DegreeOfFreedomID *id;
    
            //Unsigned int pointer to the global id of the parent node 
            unsigned* parentNodeGlobalId;
    
            //Constant reference to an enum that indicates the type of degree of freedom
            //Scalar (Temperature, concentration, etc.) or Vector component (Displacement1, Velocity1, etc.)
            DOFType const &type();
    
            double value() const;
    
            void setValue(double value);
    
            void Print();
    
        private:
            DOFType _dofType;
    
            double _value;
    };
}


#endif //UNTITLED_DEGREEOFFREEDOM_H
