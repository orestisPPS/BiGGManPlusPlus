//
// Created by hal9000 on 3/13/23.
//

#include "SteadyStateFiniteDifferenceAnalysis.h"  

namespace NumericalAnalysis {
    
    SteadyStateFiniteDifferenceAnalysis::SteadyStateFiniteDifferenceAnalysis(
            shared_ptr<SteadyStateMathematicalProblem> mathematicalProblem,
            shared_ptr<Mesh> mesh,
            shared_ptr<Solver> solver,
            shared_ptr<FDSchemeSpecs> schemeSpecs, CoordinateType coordinateSystem) :
            FiniteDifferenceAnalysis(mathematicalProblem, std::move(mesh), std::move(solver), std::move(schemeSpecs), coordinateSystem),
            steadyStateMathematicalProblem(std::move(mathematicalProblem)){
        
        auto linearSystemInitializer = make_unique<EquilibriumLinearSystemBuilder>(
                degreesOfFreedom, this->mesh, steadyStateMathematicalProblem, this->schemeSpecs, this->coordinateSystem);
        linearSystemInitializer->assembleSteadyStateLinearSystem();
        this->linearSystem = make_shared<LinearSystem>(linearSystemInitializer->K, linearSystemInitializer->RHS);
        this->solver->setLinearSystem(linearSystem);
    }
    
    void SteadyStateFiniteDifferenceAnalysis::solve() {
        solver->solve();
    }
    
    //void SteadyStateFiniteDifferenceAnalysis::createLinearSystem() {
} // NumericalAnalysis