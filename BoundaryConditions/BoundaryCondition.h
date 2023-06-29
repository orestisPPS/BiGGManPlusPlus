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
        explicit BoundaryCondition(BoundaryConditionType bcType, shared_ptr<map<DOFType,
                                   function<double (shared_ptr<vector<double>>)>>> bcForDof);
        
        //Only for double bc
        explicit BoundaryCondition(BoundaryConditionType bcType, map<DOFType, double>* bcForDof);
        
        //Returns the double value of the boundary condition for the given degree of freedom
        //at the given boundary node coordinates vector pointer.
        double getBoundaryConditionValueAtCoordinates(DOFType type, const shared_ptr<vector<double>> &coordinates);
        
        //Returns the double value of the boundary condition for the given degree of freedom
        double getBoundaryConditionValue(DOFType type);
        
        //Returns the vector value of all boundary conditions for all degrees of freedom 
        //at the given boundary node coordinates vector pointer.
        vector<double> getAllBoundaryConditionValuesAtCoordinates(const shared_ptr<vector < double>>&coordinates);
        
        //Returns an BoundaryConditionType enum constant reference of the type of the boundary condition
        const BoundaryConditionType& type() const;
        

    private:
        BoundaryConditionType _bcType;

        shared_ptr<map<DOFType, function<double (shared_ptr<vector<double>>)>>> _bcFunctionForDof;
        
        shared_ptr<map<DOFType, double>> _bcValueForDof;
    };
} // BoundaryConditions

#endif //UNTITLED_BOUNDARYCONDITION_H
