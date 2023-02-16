//
// Created by hal9000 on 2/16/23.
//

#ifndef UNTITLED_BOUNDARYCONDITION_H
#define UNTITLED_BOUNDARYCONDITION_H

#include <vector>
#include <map>
#include <functional>
#include "../PositioningInSpace/DirectionsPositions.h"
using namespace PositioningInSpace;
using namespace std;
 
namespace BoundaryConditions {

    class BoundaryCondition {
    public:
        explicit BoundaryCondition(function<double (vector<double>*)> BCFunction);

        explicit BoundaryCondition(map<Direction, function<double (vector<double>*)>> directionalBCFunction);

        double valueAt(vector<double> *coordinates);

        double valueAt(Direction direction, vector<double> *coordinates);

    private:
        function<double (vector<double>*)> _boundaryConditionFunction;

        map<Direction, function<double (vector<double>*)>> _directionalBoundaryConditionFunction;

        void _checkDirectionalBoundaryConditionFunction();
    };
} // BoundaryConditions

#endif //UNTITLED_BOUNDARYCONDITION_H
