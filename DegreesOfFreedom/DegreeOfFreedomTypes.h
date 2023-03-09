//
// Created by hal9000 on 2/19/23.
//

#ifndef UNTITLED_DEGREEOFFREEDOMTYPES_H
#define UNTITLED_DEGREEOFFREEDOMTYPES_H

#include <vector>


namespace DegreesOfFreedom {

    enum DOFType {
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
        public:
            std::vector<DOFType*>* DegreesOfFreedom;
            
            void deallocate(){
                for (auto dof : *DegreesOfFreedom){
                    delete dof;
                    dof = nullptr;
                }
                delete DegreesOfFreedom;
                DegreesOfFreedom = nullptr;
            }
    };
    
    // Contains the u component of the [1x1] displacement vector field
    struct DisplacementVectorField1D_DOFType : public Field_DOFType {
        DisplacementVectorField1D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(1);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Displacement1);
        }
    };

    // Contains the u and v components of the [2x1] displacement vector field
    // And the θ rotation (around axis 3)
    struct DisplacementVectorField2D_DOFType : public Field_DOFType {
        DisplacementVectorField2D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(3);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Displacement1);
            DegreesOfFreedom->at(1) =new DOFType(DOFType::Displacement2);
            DegreesOfFreedom->at(2) =new DOFType(DOFType::Rotation1);
        }
    };

    // Contains the u, v, and w components of the [3x1] displacement vector field
    // And the θ, ϕ, and ψ rotations (around axis 3, 2, and 1 respectively)
    struct DisplacementVectorField3D_DOFType : public Field_DOFType {
        DisplacementVectorField3D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(6);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Displacement1);
            DegreesOfFreedom->at(1) =new DOFType(DOFType::Displacement2);
            DegreesOfFreedom->at(2) =new DOFType(DOFType::Displacement3);
            DegreesOfFreedom->at(3) =new DOFType(DOFType::Rotation1);
            DegreesOfFreedom->at(4) =new DOFType(DOFType::Rotation2);
            DegreesOfFreedom->at(5) =new DOFType(DOFType::Rotation3);
        }
    };
    

    // Contains the scalar temperature T
    struct TemperatureScalar_DOFType : public Field_DOFType {
        TemperatureScalar_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(1);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Temperature);
        }
    };

    // Contains the u component of the [1x1] velocity vector field
    struct VelocityVectorField1D_DOFType : public Field_DOFType {
        VelocityVectorField1D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(1);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Velocity1);
        }
    };

    // Contains the u and v components of the [2x1] velocity vector field
    struct VelocityVectorField2D_DOFType : public Field_DOFType {
        VelocityVectorField2D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(2);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Velocity1);
            DegreesOfFreedom->at(1) =new DOFType(DOFType::Velocity2);
        }
    };
    
    // Contains the u, v, and w components of the [3x1] velocity vector field
    struct VelocityVectorField3D_DOFType : public Field_DOFType {
        VelocityVectorField3D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(3);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Velocity1);
            DegreesOfFreedom->at(1) =new DOFType(DOFType::Velocity2);
            DegreesOfFreedom->at(2) =new DOFType(DOFType::Velocity3);
        }
    };


    // Contains the p1 component of the [1x1] pressure vector field
    struct PressureVectorField1D_DOFType : public Field_DOFType {
        PressureVectorField1D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(1);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Pressure1);
        }
    };

    // Contains the p1 and p2 components of the [2x1] pressure vector field
    struct PressureVectorField2D_DOFType : public Field_DOFType {
        PressureVectorField2D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(2);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Pressure1);
            DegreesOfFreedom->at(1) =new DOFType(DOFType::Pressure2);
        }
    };

    // Contains the p1, p2, and p3 components of the [3x1] pressure vector field
    struct PressureVectorField3D_DOFType : public Field_DOFType {
        PressureVectorField3D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(3);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Pressure1);
            DegreesOfFreedom->at(1) =new DOFType(DOFType::Pressure2);
            DegreesOfFreedom->at(2) =new DOFType(DOFType::Pressure3);
        }
    };

    // Contains the scalar unknown variable
    struct UnknownScalar_DOFType : public Field_DOFType {
        UnknownScalar_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(1);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::UnknownScalarVariable);
        }
    };

    // Contains the u component of the [1x1] unknown vector field
    struct UnknownVectorField1D_DOFType : public Field_DOFType {
        UnknownVectorField1D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(1);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::UnknownVectorFieldVariableComponent1);
        }
    };

    // Contains the u and v components of the [2x1] unknown vector field
    struct UnknownVectorField2D_DOFType : public Field_DOFType {
        UnknownVectorField2D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(2);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::UnknownVectorFieldVariableComponent1);
            DegreesOfFreedom->at(1) =new DOFType(DOFType::UnknownVectorFieldVariableComponent2);
        }
    };

    // Contains the u, v, and w components of the [3x1] unknown vector field
    struct UnknownVectorField3D_DOFType : public Field_DOFType {
        UnknownVectorField3D_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(3);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::UnknownVectorFieldVariableComponent1);
            DegreesOfFreedom->at(1) =new DOFType(DOFType::UnknownVectorFieldVariableComponent2);
            DegreesOfFreedom->at(2) =new DOFType(DOFType::UnknownVectorFieldVariableComponent3);
        }
    };
    // Coupled thermo-mechanical field
    // Contains the u, v, and w components of the [3x1] displacement vector,
    // the θ, ϕ, and ψ rotations (around axis 3, 2, and 1 respectively),
    // and the scalar temperature T
    struct DisplacementVectorField3DTemperature_DOFType : public Field_DOFType {
        DisplacementVectorField3DTemperature_DOFType() : Field_DOFType{} {
            DegreesOfFreedom = new std::vector<DOFType*>(7);
            DegreesOfFreedom->at(0) =new DOFType(DOFType::Displacement1);
            DegreesOfFreedom->at(1) =new DOFType(DOFType::Displacement2);
            DegreesOfFreedom->at(2) =new DOFType(DOFType::Displacement3);
            DegreesOfFreedom->at(3) =new DOFType(DOFType::Rotation1);
            DegreesOfFreedom->at(4) =new DOFType(DOFType::Rotation2);
            DegreesOfFreedom->at(5) =new DOFType(DOFType::Rotation3);
            DegreesOfFreedom->at(6) =new DOFType(DOFType::Temperature);
        }
    };
    
} // DegreeOfFreedom

#endif //UNTITLED_DEGREEOFFREEDOMTYPES_H
