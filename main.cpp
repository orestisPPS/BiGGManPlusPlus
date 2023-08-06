#include <iostream>
#include "DegreesOfFreedom/DegreeOfFreedom.h"
#include "PartialDifferentialEquations/SecondOrderLinearPDEProperties.h"
#include "Tests/SteadyState3DNeumann.h"
#include "Tests/SteadyStateDirichlet3D.h"
#include "Tests/LanczosEigenDecompositionTest.h"
#include "Tests/QRTest.h"
#include "Tests/OperationsCUDA.h"
#include "Tests/VectorOperationsTest.h"
#include "StructuredMeshGeneration/MeshTest2D.h"
#include "BoundaryConditions/DomainBoundaryConditions.h"
#include "DegreesOfFreedom/DegreeOfFreedomTypes.h"
#include "Analysis/FiniteDifferenceAnalysis/StStFDTest.h"

#include "LinearAlgebra/Array/DecompositionMethods/DecompositionLUP.h"
#include "LinearAlgebra/FiniteDifferences/FDWeightCalculator.h"
#include <functional>
#include <list>
using namespace PartialDifferentialEquations;
using namespace LinearAlgebra;
using namespace DegreesOfFreedom;



int main() {

    //auto analysisTest = new NumericalAnalysis::StStFDTest();
    //auto neumannTest = new Tests::SteadyState3DNeumann();
    //delete neumannTest;

/*    auto lanczosTest = new Tests:: LanczosEigenDecompositionTest();
    delete lanczosTest;*/

    auto qrTest = new Tests::QRTest();
    
    //auto dirichletTest = new Tests::SteadyStateDirichlet3D();
    //auto cudaOperations = new Tests::OperationsCUDA();
    //auto singleThreadVectorOperations = new Tests::VectorOperationsTest();
    
    cout << "y000000o" << endl;
    
}



    

