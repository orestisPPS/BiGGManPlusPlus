//
// Created by hal9000 on 3/13/23.
//

#include "SteadyStateFiniteDifferenceAnalysis.h"
#include "../../LinearAlgebra/LinearSystem.h"

namespace NumericalAnalysis {
    
    SteadyStateFiniteDifferenceAnalysis::SteadyStateFiniteDifferenceAnalysis(
        SteadyStateMathematicalProblem *mathematicalProblem, Mesh *mesh, FDSchemeSpecs *schemeSpecs) :
        FiniteDifferenceAnalysis(mathematicalProblem, mesh, schemeSpecs){
        degreesOfFreedom = initiateDegreesOfFreedom();
        auto linearSystem = new LinearSystem(degreesOfFreedom, mesh);
        linearSystem->createLinearSystem();
        
        auto fileName = "linearSystem.m";
        auto filePath = "/home/hal9000/code/BiGGMan++/Testing/";
        Utility::Exporters::exportLinearSystemToMatlabFile(linearSystem->matrix, linearSystem->RHS, filePath, fileName);
    }
    
    //void SteadyStateFiniteDifferenceAnalysis::createLinearSystem() {
} // NumericalAnalysis