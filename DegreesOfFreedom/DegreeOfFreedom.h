//
// Created by hal9000 on 11/28/22.
//

#ifndef UNTITLED_DEGREEOFFREEDOM_H
#define UNTITLED_DEGREEOFFREEDOM_H

#include <tuple>
#include <iostream>
#include "limits"
#include "DegreeOfFreedomTypes.h"

using namespace std;
using namespace DegreesOfFreedom;


namespace DegreesOfFreedom{
    
    enum ConstraintType{
        Fixed,
        Free
    };

    class DegreeOfFreedom {
    
    public:
        //Use this constructor for all DOFType (fixed, free). Degree of freedom is initialized with 0 value.
        //value is initialized with NaN
        DegreeOfFreedom(DOFType* dofType, unsigned* parentNode, bool isConstrained, double value = numeric_limits<double>::quiet_NaN());
        
        bool operator == (const DegreeOfFreedom& dof);
        
        bool operator != (const DegreeOfFreedom& dof);
        
        //Constant reference to an enum that indicates the type of degree of freedom
        //Scalar (Temperature, concentration, etc.) or Vector component (Displacement1, Velocity1, etc.)
        
        unsigned int const &ID() const;
        
        DOFType const &type() const;
        
        ConstraintType const &constraintType() const;
        
        double const &value() const;
        
        unsigned int* const &parentNode() const;
        
        void setValue(double value);
        
        void setID(unsigned int ID);
        
        void print(bool printValue);
    
    private:
        
        unsigned int _id;
        
        DOFType* _dofType;
        
        ConstraintType _constraintType;
        
        double _value;
        
        unsigned int* _parentNode;
    
    };
}
#endif //UNTITLED_DEGREEOFFREEDOM_H
