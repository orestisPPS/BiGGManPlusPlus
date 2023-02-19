//
// Created by hal9000 on 2/19/23.
//

#ifndef UNTITLED_DEGREEOFFREEDOMTYPES_H
#define UNTITLED_DEGREEOFFREEDOMTYPES_H

#include <map>
#include <list>
#include "DegreeOfFreedom.h"



namespace DegreesOfFreedom {

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
    
    struct Field_DOFType{
        enum class components;
        unsigned int numberOfComponents;
    };
    
    // Contains the u component of the [1x1] displacement vector field
    struct DisplacementVectorField1D_DOFType : public Field_DOFType {
        enum class components {
            Displacement1 = static_cast<int>(DOFType::Displacement1),
        };
        DisplacementVectorField1D_DOFType() : Field_DOFType{1} {
        //TODO: Add a check to make sure that the number of components is 1    
        }
    };

    // Contains the u and v components of the [2x1] displacement vector field
    // And the θ rotation (around axis 3)
    struct DisplacementVectorField2D_DOFType : public Field_DOFType {
        enum class components {
            Displacement1 = static_cast<int>(DOFType::Displacement1),
            Displacement2 = static_cast<int>(DOFType::Displacement2),
            Rotation1 = static_cast<int>(DOFType::Rotation1),
        };
        DisplacementVectorField2D_DOFType() : Field_DOFType{3} {
        }
    };

    // Contains the u, v, and w components of the [3x1] displacement vector field
    // And the θ, ϕ, and ψ rotations (around axis 3, 2, and 1 respectively)
    struct DisplacementVectorField3D_DOFType : public Field_DOFType {
        enum class components {
            Displacement1 = static_cast<int>(DOFType::Displacement1),
            Displacement2 = static_cast<int>(DOFType::Displacement2),
            Displacement3 = static_cast<int>(DOFType::Displacement3),
            Rotation1 = static_cast<int>(DOFType::Rotation1),
            Rotation2 = static_cast<int>(DOFType::Rotation2),
            Rotation3 = static_cast<int>(DOFType::Rotation3),
        };
        DisplacementVectorField3D_DOFType() : Field_DOFType{6} {
        }
    };
    

    // Contains the scalar temperature T
    struct TemperatureScalar_DOFType : public Field_DOFType {
        enum class components {
            Temperature = static_cast<int>(DOFType::Temperature),
        };
        TemperatureScalar_DOFType() : Field_DOFType{1} {
        }
    };

    // Contains the u component of the [1x1] velocity vector field
    struct VelocityVectorField1D_DOFType : public Field_DOFType {
        enum class components {
            Velocity1 = static_cast<int>(DOFType::Velocity1),
        };
        VelocityVectorField1D_DOFType() : Field_DOFType{1} {
        }
    };

    // Contains the u and v components of the [2x1] velocity vector field
    struct VelocityVectorField2D_DOFType : public Field_DOFType {
        enum class components {
            Velocity1 = static_cast<int>(DOFType::Velocity1),
            Velocity2 = static_cast<int>(DOFType::Velocity2),
        };
        VelocityVectorField2D_DOFType() : Field_DOFType{2} {
        }
    };
    
    // Contains the u, v, and w components of the [3x1] velocity vector field
    struct VelocityVectorField3D_DOFType : public Field_DOFType {
        enum class components {
            Velocity1 = static_cast<int>(DOFType::Velocity1),
            Velocity2 = static_cast<int>(DOFType::Velocity2),
            Velocity3 = static_cast<int>(DOFType::Velocity3),
        };
        VelocityVectorField3D_DOFType() : Field_DOFType{3} {
        }
    };


    // Contains the p1 component of the [1x1] pressure vector field
    struct PressureVectorField1D_DOFType : public Field_DOFType {
        enum class components {
            Pressure1 = static_cast<int>(DOFType::Pressure1),
        };
        PressureVectorField1D_DOFType() : Field_DOFType{1} {
        }
    };

    // Contains the p1 and p2 components of the [2x1] pressure vector field
    struct PressureVectorField2D_DOFType : public Field_DOFType {
        enum class components {
            Pressure1 = static_cast<int>(DOFType::Pressure1),
            Pressure2 = static_cast<int>(DOFType::Pressure2),
        };
        PressureVectorField2D_DOFType() : Field_DOFType{2} {
        }
    };

    // Contains the p1, p2, and p3 components of the [3x1] pressure vector field
    struct PressureVectorField3D_DOFType : public Field_DOFType {
        enum class components {
            Pressure1 = static_cast<int>(DOFType::Pressure1),
            Pressure2 = static_cast<int>(DOFType::Pressure2),
            Pressure3 = static_cast<int>(DOFType::Pressure3),
        };
        PressureVectorField3D_DOFType() : Field_DOFType{3} {
        }
    };

    // Contains the scalar unknown variable
    struct UnknownScalar_DOFType : public Field_DOFType {
        enum class components {
            UnknownScalarVariable = static_cast<int>(DOFType::UnknownScalarVariable),
        };
        UnknownScalar_DOFType() : Field_DOFType{1} {
        }
    };

    // Contains the u component of the [1x1] unknown vector field
    struct UnknownVectorField1D_DOFType : public Field_DOFType {
        enum class components {
            UnknownVectorFieldVariableComponent1 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent1),
        };
        UnknownVectorField1D_DOFType() : Field_DOFType{1} {
        }
    };

    // Contains the u and v components of the [2x1] unknown vector field
    struct UnknownVectorField2D_DOFType : public Field_DOFType {
        enum class components {
            UnknownVectorFieldVariableComponent1 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent1),
            UnknownVectorFieldVariableComponent2 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent2),
        };
        UnknownVectorField2D_DOFType() : Field_DOFType{2} {
        }
    };

    // Contains the u, v, and w components of the [3x1] unknown vector field
    struct UnknownVectorField3D_DOFType : public Field_DOFType {
        enum class components {
            UnknownVectorFieldVariableComponent1 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent1),
            UnknownVectorFieldVariableComponent2 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent2),
            UnknownVectorFieldVariableComponent3 = static_cast<int>(DOFType::UnknownVectorFieldVariableComponent3),
        };
        UnknownVectorField3D_DOFType() : Field_DOFType{3} {
        }
    };
    
} // DegreeOfFreedom

#endif //UNTITLED_DEGREEOFFREEDOMTYPES_H
