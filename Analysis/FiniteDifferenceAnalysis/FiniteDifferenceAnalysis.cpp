//
// Created by hal9000 on 1/31/23.
//

#include "FiniteDifferenceAnalysis.h"

#include <utility>

namespace NumericalAnalysis {
    
    FiniteDifferenceAnalysis::FiniteDifferenceAnalysis(shared_ptr<MathematicalProblem>problem,
                                                       shared_ptr<Mesh> mesh,
                                                       shared_ptr<Solver> solver,
                                                       shared_ptr<FDSchemeSpecs> specs,
                                                       CoordinateType coordinateSystem) :
                                schemeSpecs(std::move(specs)),
                                NumericalAnalysis(std::move(problem), std::move(mesh), std::move(solver) ) {
    }
    

} // NumericalAnalysis