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
#include "LinearAlgebra/FiniteDifferences/FDWeightCalculator.h"
#include <functional>
#include <list>
int main() {

/*    auto test = new StructuredMeshGenerator::MeshTest2D();
    auto analysisTest = new NumericalAnalysis::StStFDTest();*/

    const unsigned max_deriv = 2;
    std::vector<std::string> labels {"0th derivative (interpolation)", "1st derivative", "2nd derivative"};
    //std::vector<double> x {0, 1, -1, 2, -2};  // Fourth order of accuracy
    std::vector<double> x {1,2,3};  // Fourth order of accuracy
    std::vector<double> f {2,2,2};
    auto coeffs = LinearAlgebra::calculateWeights(x, max_deriv, 2.0);
    auto values = LinearAlgebra::calculateDerivatives(x, f, max_deriv, 2.0);
    for (unsigned deriv_i = 0; deriv_i <= max_deriv; deriv_i++){
        std::cout << labels[deriv_i] << ": ";
        for (unsigned idx = 0; idx < x.size(); idx++){
            std::cout << values[deriv_i][idx] << " ";
        }
        std::cout << std::endl;
    }
    
    
    
    cout << "MTSTK GMS" << endl;
    
}



    

