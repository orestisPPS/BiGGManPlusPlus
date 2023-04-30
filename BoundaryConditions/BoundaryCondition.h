//
// Created by hal9000 on 2/16/23.
//

#ifndef UNTITLED_BOUNDARYCONDITION_H
#define UNTITLED_BOUNDARYCONDITION_H

#include <vector>
#include <map>
#include <functional>
#include "../PositioningInSpace/DirectionsPositions.h"
#include "../DegreesOfFreedom/DegreeOfFreedomTypes.h"

using namespace PositioningInSpace;
using namespace DegreesOfFreedom;
using namespace std;
 
namespace BoundaryConditions {

    enum BoundaryConditionType {
        Dirichlet,
        Neumann
    };
    
    class BoundaryCondition {
    public:
        //Boundary Condition for all degrees of freedom of the problem defined for a single boundary position
        explicit BoundaryCondition(BoundaryConditionType bcType, map<DOFType, function<double (vector<double>*)>>* bcForDof);
        
        //Only for double bc
        explicit BoundaryCondition(BoundaryConditionType bcType, map<DOFType, double>* bcForDof);
        
        //Returns the double value of the boundary condition for the given degree of freedom
        //at the given boundary node coordinates vector pointer.
        double scalarValueOfDOFAt(DOFType type, vector<double>* coordinates);
        
        //Returns the vector value of all boundary conditions for all degrees of freedom 
        //at the given boundary node coordinates vector pointer.
        vector<double> vectorValueOfAllDOFAt(vector<double> *coordinates);
        
        //Returns an BoundaryConditionType enum constant reference of the type of the boundary condition
        const BoundaryConditionType& type() const;
        

    private:
        BoundaryConditionType _bcType;

        map<DOFType, function<double (vector<double>*)>>* bcForDof;
    };
} // BoundaryConditions

#endif //UNTITLED_BOUNDARYCONDITION_H
