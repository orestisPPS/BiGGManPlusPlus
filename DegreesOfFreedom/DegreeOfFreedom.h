//
// Created by hal9000 on 11/28/22.
//

#ifndef UNTITLED_DEGREEOFFREEDOM_H
#define UNTITLED_DEGREEOFFREEDOM_H

#include <tuple>
#include "DegreeOfFreedomID.h"
using namespace std;


namespace DegreesOfFreedom{
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
    

    class DegreeOfFreedom {
    public:

        explicit DegreeOfFreedom(DOFType dofType);

        DegreeOfFreedom(DOFType dofType, double value);
        
        bool operator == (const DegreeOfFreedom& dof);

        bool operator != (const DegreeOfFreedom& dof);
        
        DegreeOfFreedomID id;
        
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
