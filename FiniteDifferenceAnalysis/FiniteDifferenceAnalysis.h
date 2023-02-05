//
// Created by hal9000 on 1/31/23.
//

#ifndef UNTITLED_FINITEDIFFERENCEANALYSIS_H
#define UNTITLED_FINITEDIFFERENCEANALYSIS_H

#include "../LinearAlgebra/FiniteDifferences/FDSchemeSpecs.h"
#include "../MathematicalProblem/SteadyStateMathematicalProblem.h"
#include "../MathematicalProblem/TransientMathematicalProblem.h"
#include "../Discretization/Mesh/Mesh.h"

using namespace LinearAlgebra;
using namespace MathematicalProblem;
using namespace Discretization;

namespace NumericalAnalysis {

    class FiniteDifferenceAnalysis {
    public:
        FiniteDifferenceAnalysis(SteadyStateMathematicalProblem &mathematicalProblem,
                                 Mesh *mesh,
                                 FDSchemeSpecs &schemeSpecs);

    };

} // NumericalAnalysis

#endif //UNTITLED_FINITEDIFFERENCEANALYSIS_H
