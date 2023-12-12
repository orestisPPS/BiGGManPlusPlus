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



int main() {
    //NUMERICAL VECTOR AND MATRIX TESTS.
    //Tests::NumericalVectorTest::runTests();
    //Tests::NumericalMatrixTest::runTests();
    
    //ANALYSIS TESTS.
    //Tests::SteadyStateDirichlet2D::runTests(); //Pass
    //Tests::TransientDirichlet2D::runTests(); //Fail
    Tests::SteadyState3DDirichlet::runTests(); //Pass
    //Tests::SteadyState3DNeumann::runTests(); //Pass
    //Tests::TransientDirichlet3D::runTests(); //Fail
    

/*    auto lanczosTest = new Tests:: LanczosEigenDecompositionTest();
    delete lanczosTest;*/

    //Tests::IterativeSolversTest::runTests();
    //Tests::TransientDirichlet3D::runTests();
    //Tests::TransientDirichlet3D::runTests();
    //Tests::SteadyState3DNeumann::runTests();
    //Tests::MeshGenerator::buildMesh();


    

    //auto cudaOperations = new Tests::OperationsCUDA();
    //auto singleThreadVectorOperations = new Tests::VectorOperationsTest();

    cout << "That's All Folks!" << endl;

}



    

