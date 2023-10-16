//
// Created by hal9000 on 3/13/23.
//

#include "SteadyStateFiniteDifferenceAnalysis.h"  

namespace NumericalAnalysis {
    
    SteadyStateFiniteDifferenceAnalysis::SteadyStateFiniteDifferenceAnalysis(
            const shared_ptr<SteadyStateMathematicalProblem>& mathematicalProblem, const shared_ptr<Mesh>& mesh, const shared_ptr<Solver>& solver,
            const shared_ptr<FDSchemeSpecs>&schemeSpecs, CoordinateType coordinateSystem) :
            FiniteDifferenceAnalysis(mathematicalProblem, mesh, solver, schemeSpecs, coordinateSystem) {
        
        auto linearSystemInitializer =
                make_unique<AnalysisLinearSystemInitializer>(degreesOfFreedom, mesh, mathematicalProblem, schemeSpecs, coordinateSystem);
        linearSystemInitializer->createLinearSystem();
        this->linearSystem = linearSystemInitializer->linearSystem;
        solver->setLinearSystem(linearSystem);
    }
    
    void SteadyStateFiniteDifferenceAnalysis::solve() const {
        solver->solve();
    }
    
    //void SteadyStateFiniteDifferenceAnalysis::createLinearSystem() {
} // NumericalAnalysis