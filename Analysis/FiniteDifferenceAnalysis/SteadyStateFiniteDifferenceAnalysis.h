//
// Created by hal9000 on 3/13/23.
//

#ifndef UNTITLED_STEADYSTATEFINITEDIFFERENCEANALYSIS_H
#define UNTITLED_STEADYSTATEFINITEDIFFERENCEANALYSIS_H

#include "FiniteDifferenceAnalysis.h"
#include "../../MathematicalEntities/MathematicalProblem/SteadyStateMathematicalProblem.h"

namespace NumericalAnalysis {

    class SteadyStateFiniteDifferenceAnalysis : public FiniteDifferenceAnalysis{
    public:
        
        SteadyStateFiniteDifferenceAnalysis(const shared_ptr<SteadyStateMathematicalProblem>& mathematicalProblem,
                                            const shared_ptr<Mesh>& mesh,
                                            const shared_ptr<Solver>& solver,
                                            const shared_ptr<FDSchemeSpecs>&schemeSpecs, CoordinateType coordinateSystem = Natural);
        
        void solve() const override;
    };

} // NumericalAnalysis

#endif //UNTITLED_STEADYSTATEFINITEDIFFERENCEANALYSIS_H
