//
// Created by hal9000 on 11/28/22.
//

#ifndef UNTITLED_DEGREEOFFREEDOM_H
#define UNTITLED_DEGREEOFFREEDOM_H

#include <tuple>
using namespace std;


namespace DegreesOfFreedom{
    enum DOFType {
        X,
        Y,
        Z,
        Temperature,
        Pressure,
        Displacement,
        Rotation,
        Velocity,
        UnknownVariable
    };

    enum FieldType {
        Scalar,
        VectorComponent1,
        VectorComponent2,
        VectorComponent3,
    };

    class DegreeOfFreedom {
    public:

        DegreeOfFreedom(DOFType dofType, FieldType fieldType);

        DegreeOfFreedom(DOFType dofType, FieldType fieldType, double value);

        ~DegreeOfFreedom();

        DOFType type();

        FieldType fieldType();

        double value();

        void setValue(double value);

        bool operator == (const DegreeOfFreedom& dof);

        bool operator != (const DegreeOfFreedom& dof);

        void Print();

    private:
        DOFType _dofType;
        FieldType _fieldType;
        double *_value;
    };
}


#endif //UNTITLED_DEGREEOFFREEDOM_H
