//
// Created by hal9000 on 11/29/22.
//

#ifndef UNTITLED_BOUNDARYCONDITION_H
#define UNTITLED_BOUNDARYCONDITION_H

#include <vector>
#include <map>
#include <functional>
#include <list>
#include "../PositioningInSpace/DirectionsPositions.h"
using namespace PositioningInSpace;
using namespace std;
namespace BoundaryConditions {

    class BoundaryCondition {
    public:

        BoundaryCondition(function<double (vector<double>)> *BCFunction);

        BoundaryCondition(list<tuple<PositioningInSpace::Direction, function<double (vector<double>)>*>> directionalBCFunction);

        ~BoundaryCondition();

        function<double (vector<double>)> *boundaryConditionFunction;

        list<tuple<PositioningInSpace::Direction, function<double (vector<double>)>*>> directionalBoundaryConditionFunction;


        
    };

} // BoundaryConditions

#endif //UNTITLED_BOUNDARYCONDITION_H
