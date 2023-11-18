#include <iostream>
#include "Tests/AnalysisTests/SteadyState3DNeumann/SteadyState3DNeumann.h"
#include "Tests/AnalysisTests/SteadyStateDirichlet2D/SteadyStateDirichlet2D.h"
#include "Tests/AnalysisTests/SteadyState3DDirichlet/SteadyState3DDirichlet.h"
#include "Tests/AnalysisTests/Transient2DDirichlet/TransientDirichlet2D.h"
#include "Tests/AnalysisTests/TransientDirichlet3D/TransientDirichlet3D.h"
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

    //Tests::SteadyStateDirichlet2D::runTests(); //Pass
    //Tests::TransientDirichlet2D::runTests(); //Pass
    Tests::SteadyState3DDirichlet::runTests(); //Pass
    //Tests::SteadyState3DNeumann::runTests(); //Fail
    //Tests::TransientDirichlet3D::runTests(); //Fail
    

/*    auto lanczosTest = new Tests:: LanczosEigenDecompositionTest();
    delete lanczosTest;*/

    //Tests::IterativeSolversTest::runTests();
    //Tests::SteadyStateDirichlet2D::runTests();
    //Tests::TransientDirichlet3D::runTests();
    //Tests::TransientDirichlet3D::runTests();
    //Tests::SteadyState3DNeumann::runTests();
    //Tests::MeshGenerator::buildMesh();
/*    auto vectorTest = new NumericalVectorTest();
    vectorTest->runTests();
    Tests::NumericalMatrixTest::runTests();*/


    //auto qrTest = new Tests::QRTest();

    //auto dirichletTest = new Tests::SteadyStateDirichlet3D();
    
    Tests::SteadyState3DDirichlet::runTests();

    //Tests::SteadyState3DNeumann::runTests();

    //auto cudaOperations = new Tests::OperationsCUDA();
    //auto singleThreadVectorOperations = new Tests::VectorOperationsTest();

    cout << "y000000o" << endl;

}



    

