//
// Created by hal9000 on 1/31/23.
//

#include "FiniteDifferenceAnalysis.h"

namespace NumericalAnalysis {
    
    FiniteDifferenceAnalysis::FiniteDifferenceAnalysis(shared_ptr<MathematicalProblem> mathematicalProblem,
                                                       shared_ptr<Mesh> mesh, shared_ptr<Solver> solver,
                                                       shared_ptr<FDSchemeSpecs> schemeSpecs, CoordinateType coordinateSystem) :
            NumericalAnalysis(std::move(mathematicalProblem), std::move(mesh), std::move(solver), coordinateSystem),
            schemeSpecs(std::move(schemeSpecs)) {
    }
    

} // NumericalAnalysis