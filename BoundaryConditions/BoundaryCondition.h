//
// Created by hal9000 on 2/16/23.
//

#ifndef UNTITLED_BOUNDARYCONDITION_H
#define UNTITLED_BOUNDARYCONDITION_H

#include <vector>
#include <map>
#include <functional>
#include "../PositioningInSpace/DirectionsPositions.h"
#include "DomainBoundaryConditions.h"

using namespace PositioningInSpace;
using namespace std;
 
namespace BoundaryConditions {

    enum BoundaryConditionType {
        Dirichlet,
        Neumann
    };
    class BoundaryCondition {
    public:
        
        explicit BoundaryCondition(BoundaryConditionType bcType, map<DOFType*, function<double (vector<double>*)>>* bcForDof);
        
        double scalarValueOfDOFAt(DOFType type, vector<double> *coordinates);
        
        vector<double> vectorValueOfAllDOFAt(vector<double> *coordinates);
        
        const BoundaryConditionType& type() const;
        

    private:
        BoundaryConditionType _bcType;

        map<DOFType*, function<double (vector<double>*)>>* bcForDof;
    };
} // BoundaryConditions

#endif //UNTITLED_BOUNDARYCONDITION_H
