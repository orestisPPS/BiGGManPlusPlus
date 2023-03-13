//
// Created by hal9000 on 1/31/23.
//

#include "FiniteDifferenceAnalysis.h"

namespace NumericalAnalysis {
    
    FiniteDifferenceAnalysis::FiniteDifferenceAnalysis(MathematicalProblem *problem, Mesh *mesh, FDSchemeSpecs *specs) :
            schemeSpecs(specs), NumericalAnalysis(problem, mesh) {
    }
    

} // NumericalAnalysis