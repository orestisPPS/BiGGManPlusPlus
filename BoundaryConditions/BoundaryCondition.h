//
// Created by hal9000 on 11/29/22.
//

#include <vector>
#include <map>
#include <functional>

#include "../PositioningInSpace/DirectionsPositions.h"
using namespace PositioningInSpace;
using namespace std;
namespace BoundaryConditions {

    class BoundaryCondition {
    public:
        
        // auto firstBCBoi = new std::function<double(vector<double>)>([](vector<double> x){return x[0] + x[1];});
        //    auto testVector = new vector<double>();
        //    testVector->push_back(1.0);
        //    testVector->push_back(8.0);
        //    
        //    std::cout << (*firstBCBoi)(*testVector) << std::endl;
        
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


