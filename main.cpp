#include <iostream>

#include "DegreesOfFreedom/DegreeOfFreedom.h"
int main() {

    auto firstDofBoi = new DegreeOfFreedom(DOFType::Temperature, FieldType::VectorComponent1, 1.0);
    firstDofBoi->Print();
    return 0;
}
