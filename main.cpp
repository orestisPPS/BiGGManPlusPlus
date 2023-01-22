#include <iostream>

#include "DegreesOfFreedom/DegreeOfFreedom.h"
using namespace DegreesOfFreedom;
//#include "PositioningInSpace/Coordinate.h"
//using namespace PositioningInSpace;
#include "PartialDifferentialEquations/SecondOrderLinearPDEProperties.h"
using namespace PartialDifferentialEquations;
using namespace LinearAlgebra;
#include "Discretization/Mesh/Mesh.h"
#include "StructuredMeshGeneration/MeshTest2D.h"
#include <functional>
#include <vector>
#include <list>
#include <tuple>
int main() {
    auto matrix = Array<double>(2, 2);
    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(1, 0) = 3;
    matrix(1, 1) = 4;
    
    auto matrix2 = Array<double>(2, 2);
    matrix2(0, 0) = 1;
    matrix2(0, 1) = 2;
    matrix2(1, 0) = 3;
    matrix2(1, 1) = 8;
    
    matrix =  matrix2 * 2;
    matrix.print();
    matrix = matrix2 - matrix2;
    matrix.print();
    
    
    
    //cout << matrix.size() << endl;
    //cout << matrix.vectorElement(4) << endl;
/*    auto matrix3 = matrix + matrix2;
    auto meshTest = StructuredMeshGenerator::MeshTest2D();
    return 0;*/
}

/*    auto firstDofBoi = new DegreeOfFreedom(DOFType::Temperature, FieldType::VectorComponent1, 1.0);
    //firstDofBoi->Print();
    
    //auto firstCoordinateBoi = new Coordinate(CoordinateType::NaturalCoordinateSystem, Direction::One, 1.0);
    //firstCoordinateBoi->Print();
    
 
    auto firstBCBoi = new std::function<double(vector<double>)>([](vector<double> x){return x[0] + x[1];});
    auto testVector = new vector<double>();
    testVector->push_back(1.0);
    testVector->push_back(8.0);

    
    std::cout << (*firstBCBoi)(*testVector) << std::endl;
    std::cout << "Hello, World!" << std::endl;
       */
    

