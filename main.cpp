#include <iostream>
#include "DegreesOfFreedom/DegreeOfFreedom.h"
#include "PartialDifferentialEquations/SecondOrderLinearPDEProperties.h"
#include "Tests/SteadyState3DNeumann.h"
#include "Tests/SteadyStateDirichlet3D.h"
#include "Tests/LanczosEigenDecompositionTest.h"
#include "Tests/NumericalVectorTest.h"
#include "Tests/QRTest.h"
#include "Tests/OperationsCUDA.h"
#include "Tests/VectorOperationsTest.h"
#include "Tests/NumericalMatrixTest.h"
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

/*    auto vectorTest = new NumericalVectorTest();
    vectorTest->runTests();*/
 Tests::NumericalMatrixTest::runTests();

 
 
    //auto vectorTest = new NumericalVectorTest();
/*    vectorTest->runTests();
    
    auto numericalVector = NumericalVector<double>(10, MultiThread);
    auto numericalVectorRawPtr = new NumericalVector<double>(10, MultiThread);
    auto numericalVectorSharedPtr = make_shared<NumericalVector<double>>(10, MultiThread);
    auto numericalVectorUniquePtr = make_unique<NumericalVector<double>>(10, MultiThread);
    for (unsigned i = 0; i < 10; ++i) {
        numericalVector[i] = i;
        (*numericalVectorRawPtr)[i] = i;
        (*numericalVectorSharedPtr)[i] = i;
        (*numericalVectorUniquePtr)[i] = i;
    }

    NumericalVector<double> subtraction = NumericalVector<double>(10);
    numericalVectorSharedPtr->subtract(numericalVectorSharedPtr, subtraction);
    
    auto numericalVector2 = NumericalVector<double>(numericalVectorSharedPtr);
    auto numericalVector4 = NumericalVector<double>(numericalVector);
    
    auto numericalVector3 = NumericalVector<double>(10, MultiThread);
    numericalVector3 = numericalVector2;*/
    
    
    
    

    //auto qrTest = new Tests::QRTest();
    
    //auto dirichletTest = new Tests::SteadyStateDirichlet3D();
    //auto cudaOperations = new Tests::OperationsCUDA();
    //auto singleThreadVectorOperations = new Tests::VectorOperationsTest();
    
    cout << "y000000o" << endl;
    
}



    

