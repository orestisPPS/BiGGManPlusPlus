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
        linearSystem->createLinearSystem(mesh);
    }
    
    //void SteadyStateFiniteDifferenceAnalysis::createLinearSystem() {
} // NumericalAnalysis