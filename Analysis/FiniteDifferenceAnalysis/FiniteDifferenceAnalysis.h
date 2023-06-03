//
// Created by hal9000 on 1/31/23.
//

#ifndef UNTITLED_FINITEDIFFERENCEANALYSIS_H
#define UNTITLED_FINITEDIFFERENCEANALYSIS_H

#include "../../LinearAlgebra/FiniteDifferences/FDSchemeSpecs.h"
#include "../NumericalAnalysis.h"

using namespace LinearAlgebra;
using namespace MathematicalProblems;
using namespace Discretization;

namespace NumericalAnalysis {

    class FiniteDifferenceAnalysis : public NumericalAnalysis {
    public:
        FiniteDifferenceAnalysis(MathematicalProblem *mathematicalProblem, Mesh *mesh, Solver* solver, FDSchemeSpecs *schemeSpecs, CoordinateType coordinateSystem = Natural);
        
        FDSchemeSpecs* schemeSpecs;
       
    };

} // NumericalAnalysis

#endif //UNTITLED_FINITEDIFFERENCEANALYSIS_H
