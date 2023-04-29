#include <iostream>

#include "DegreesOfFreedom/DegreeOfFreedom.h"
using namespace DegreesOfFreedom;
#include "PartialDifferentialEquations/SecondOrderLinearPDEProperties.h"
using namespace PartialDifferentialEquations;
using namespace LinearAlgebra;
#include "StructuredMeshGeneration/MeshTest2D.h"
#include "BoundaryConditions/DomainBoundaryConditions.h"
#include "DegreesOfFreedom/DegreeOfFreedomTypes.h"
#include "Analysis/FiniteDifferenceAnalysis/StStFDTest.h"
#include "LinearAlgebra/Array/DecompositionMethods/DecompositionLUP.h"
#include <functional>
#include <list>
int main() {

    auto test = new StructuredMeshGenerator::MeshTest2D();
    auto analysisTest = new NumericalAnalysis::StStFDTest();
    
/*    unsigned int n = 3;
    auto LU = new Array<double>(n, n);
    LU->at(0, 0) = 2;
    LU->at(0, 1) = 1;
    LU->at(0, 2) = -1;
    LU->at(1, 0) = -4;
    LU->at(1, 1) = 2;
    LU->at(1, 2) = 3;
    LU->at(2, 0) = -4;
    LU->at(2, 1) = 3;
    LU->at(2, 2) = 5;
    
    auto lup = new DecompositionLUP(LU);
    lup->decompose(false);
    auto l = lup->getL();
    l->;
    cout<<"------------------" << endl;
    auto u = lup->getU();
    u->print();
    auto det = lup->determinant();
    cout << "det = " << det << endl;
    
    auto solution = lup->solve(new vector<double>(n, 1));
    for (auto i = 0; i < n; i++){
        cout << solution->at(i) << endl;
    }*/
    
    
    
    
    cout << "MTSTK GMS" << endl;
    
}



    

