#include <iostream>

#include "DegreesOfFreedom/DegreeOfFreedom.h"
using namespace DegreesOfFreedom;
//#include "PositioningInSpace/Coordinate.h"
//using namespace PositioningInSpace;
#include "PartialDifferentialEquations/SecondOrderLinearPDEProperties.h"
using namespace PartialDifferentialEquations;
#include "Primitives/Matrix.h"
using namespace Primitives;
#include <functional>
#include <vector>
#include <list>
#include <tuple>
int main() {

    auto firstDofBoi = new DegreeOfFreedom(DOFType::Temperature, FieldType::VectorComponent1, 1.0);
    //firstDofBoi->Print();
    
    //auto firstCoordinateBoi = new Coordinate(CoordinateType::Natural, Direction::One, 1.0);
    //firstCoordinateBoi->Print();
    
 
    auto firstBCBoi = new std::function<double(vector<double>)>([](vector<double> x){return x[0] + x[1];});
    auto testVector = new vector<double>();
    testVector->push_back(1.0);
    testVector->push_back(8.0);

    
    std::cout << (*firstBCBoi)(*testVector) << std::endl;
    std::cout << "Hello, World!" << std::endl;
        
    return 0;
    }
