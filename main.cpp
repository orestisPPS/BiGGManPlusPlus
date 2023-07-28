#include <iostream>

#include "DegreesOfFreedom/DegreeOfFreedom.h"
using namespace DegreesOfFreedom;
#include "PartialDifferentialEquations/SecondOrderLinearPDEProperties.h"
#include "Tests/SteadyState3DNeumann.h"
#include "Tests/OperationsCUDA.h"
using namespace PartialDifferentialEquations;
using namespace LinearAlgebra;
#include "StructuredMeshGeneration/MeshTest2D.h"
#include "BoundaryConditions/DomainBoundaryConditions.h"
#include "DegreesOfFreedom/DegreeOfFreedomTypes.h"
#include "Analysis/FiniteDifferenceAnalysis/StStFDTest.h"
#include "LinearAlgebra/Array/DecompositionMethods/DecompositionLUP.h"
#include "LinearAlgebra/FiniteDifferences/FDWeightCalculator.h"
#include <functional>
#include <list>

int main() {

    //auto analysisTest = new NumericalAnalysis::StStFDTest();
    auto neumannTest = new Tests::SteadyState3DNeumann();
    //auto cudaOperations = new Tests::OperationsCUDA();
    
    cout << "y000000o" << endl;
    
}



    

