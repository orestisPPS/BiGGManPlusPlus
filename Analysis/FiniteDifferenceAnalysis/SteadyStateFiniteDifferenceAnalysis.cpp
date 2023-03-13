//
// Created by hal9000 on 3/13/23.
//

#include "SteadyStateFiniteDifferenceAnalysis.h"

namespace NumericalAnalysis {
    
    SteadyStateFiniteDifferenceAnalysis::SteadyStateFiniteDifferenceAnalysis(
        SteadyStateMathematicalProblem *mathematicalProblem, Mesh *mesh, FDSchemeSpecs *schemeSpecs) :
        FiniteDifferenceAnalysis(mathematicalProblem, mesh, schemeSpecs){
        degreesOfFreedom = initiateDegreesOfFreedom();
    }
    
    //void SteadyStateFiniteDifferenceAnalysis::createLinearSystem() {
} // NumericalAnalysis