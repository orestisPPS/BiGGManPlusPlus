#include <iostream>

#include "DegreesOfFreedom/DegreeOfFreedom.h"
#include "PositioningInSpace/Coordinate.h"
int main() {

    auto firstDofBoi = new DegreeOfFreedom(DOFType::Temperature, FieldType::VectorComponent1, 1.0);
    firstDofBoi->Print();
    
    auto firstCoordinateBoi = new Coordinate(CoordinateType::Natural, Direction::One, 1.0);
    firstCoordinateBoi->Print();
    return 0;
}
