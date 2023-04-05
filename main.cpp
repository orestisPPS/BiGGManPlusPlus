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
#include <functional>
#include <list>
int main() {

    //auto test = new StructuredMeshGenerator::MeshTest2D();
    auto analysisTest = new NumericalAnalysis::StStFDTest();
    
    cout << "MTSTK GMS" << endl;
    
}



    

