#include <iostream>
#include "Tests/AnalysisTests/SteadyState3DNeumann.h"
#include "Tests/AnalysisTests/SteadyState2DTest.h"
#include "Tests/AnalysisTests/SteadyStateDirichlet3D.h"
#include "Tests/LanczosEigenDecompositionTest.h"
#include "Tests/NumericalVectorTest.h"
#include "Tests/IterativeSolversTest.h"
#include "Tests/QRTest.h"
#include "Tests/OperationsCUDA.h"
#include "Tests/NumericalMatrixTest.h"
#include "StructuredMeshGeneration/MeshTest2D.h"
#include "DegreesOfFreedom/DegreeOfFreedomTypes.h"
#include "Analysis/FiniteDifferenceAnalysis/StStFDTest.h"

#include "LinearAlgebra/FiniteDifferences/FDWeightCalculator.h"
#include <functional>
#include <list>
using namespace MathematicalEntities;
using namespace LinearAlgebra;
using namespace DegreesOfFreedom;



int main() {

    //auto analysisTest = new NumericalAnalysis::StStFDTest();
/*
    auto neumannTest = new Tests::SteadyState3DNeumann();
    delete neumannTest;
*/

/*    auto lanczosTest = new Tests:: LanczosEigenDecompositionTest();
    delete lanczosTest;*/

    //Tests::IterativeSolversTest::runTests();
    Tests::SteadyState2DTest::runTests();
    //Tests::SteadyState3DNeumann::runTests();
    //Tests::MeshGenerator::buildMesh();
/*    auto vectorTest = new NumericalVectorTest();
    vectorTest->runTests();
    Tests::NumericalMatrixTest::runTests();*/


    //auto qrTest = new Tests::QRTest();

    //auto dirichletTest = new Tests::SteadyStateDirichlet3D();
    //auto cudaOperations = new Tests::OperationsCUDA();
    //auto singleThreadVectorOperations = new Tests::VectorOperationsTest();

    cout << "y000000o" << endl;

}



    

