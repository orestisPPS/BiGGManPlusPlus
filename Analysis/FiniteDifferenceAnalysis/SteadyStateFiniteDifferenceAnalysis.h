//
// Created by hal9000 on 3/13/23.
//

#ifndef UNTITLED_STEADYSTATEFINITEDIFFERENCEANALYSIS_H
#define UNTITLED_STEADYSTATEFINITEDIFFERENCEANALYSIS_H

#include "FiniteDifferenceAnalysis.h"

namespace NumericalAnalysis {

    class SteadyStateFiniteDifferenceAnalysis : public FiniteDifferenceAnalysis{
    public:
        
        SteadyStateFiniteDifferenceAnalysis(SteadyStateMathematicalProblem *mathematicalProblem, Mesh *mesh,
                                            Solver *solver, FDSchemeSpecs *schemeSpecs, CoordinateType coordinateSystem = Natural);
    };

} // NumericalAnalysis

#endif //UNTITLED_STEADYSTATEFINITEDIFFERENCEANALYSIS_H
