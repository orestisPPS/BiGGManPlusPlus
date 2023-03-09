//
// Created by hal9000 on 11/28/22.
//

#ifndef UNTITLED_DEGREEOFFREEDOM_H
#define UNTITLED_DEGREEOFFREEDOM_H

#include <tuple>
#include <iostream>
#include "DegreeOfFreedomID.h"
#include "DegreeOfFreedomTypes.h"
#include "../Discretization/Node/Node.h"
using namespace Discretization;
using namespace DegreesOfFreedom;

using namespace std;
using namespace DegreesOfFreedom;


namespace DegreesOfFreedom{
    
    class DegreeOfFreedom {
        public:
            //Use this constructor when the degree of freedom is not fixed. DOFs with constraint type free or flux
            // will be initialized with a value of NaN.
            DegreeOfFreedom(DegreesOfFreedom::DOFType dofType, Node* parentNode, bool isConstrained);
    
            //Use this constructor when the degree of freedom is fixed. Only DOFs with constraint type fixed
            // can be initialized with this constructor.
            DegreeOfFreedom(DegreesOfFreedom::DOFType dofType, double value, Node* parentNode, bool isConstrained);
    
            ~DegreeOfFreedom();
    
            bool operator == (const DegreeOfFreedom& dof);
    
            bool operator != (const DegreeOfFreedom& dof);
    
            //Pointer to the id of the degree of freedom. Contains ConstraintType Enum (fixed, flux, free)
            // and unsigned int id. The enumeration of the id corresponds to the constraint type.
            DegreeOfFreedomID *id;
    
            //Unsigned int pointer to the global id of the parent node 
            Node* parentNode;
    
            //Constant reference to an enum that indicates the type of degree of freedom
            //Scalar (Temperature, concentration, etc.) or Vector component (Displacement1, Velocity1, etc.)
            DOFType type();
    
            double value() const;
    
            void setValue(double value);
    
            void Print();
    
        private:
            DOFType _dofType;
    
            double _value;
    };
}


#endif //UNTITLED_DEGREEOFFREEDOM_H
