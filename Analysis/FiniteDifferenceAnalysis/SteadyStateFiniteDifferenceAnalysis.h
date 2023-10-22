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
        
        SteadyStateFiniteDifferenceAnalysis(shared_ptr<SteadyStateMathematicalProblem> mathematicalProblem,
                                            shared_ptr<Mesh> mesh,
                                            shared_ptr<Solver> solver,
                                            shared_ptr<FDSchemeSpecs> schemeSpecs, CoordinateType coordinateSystem = Natural);
        
        shared_ptr<SteadyStateMathematicalProblem> steadyStateMathematicalProblem;
        
        void solve() override;
        
        void applySolutionToDegreesOfFreedom() const override;

        virtual NumericalVector<double> getSolutionAtNode(NumericalVector<double>& nodeCoordinates, double tolerance = 1E-4, DOFType = NoDOFType) const override;

    };

} // NumericalAnalysis

#endif //UNTITLED_STEADYSTATEFINITEDIFFERENCEANALYSIS_H
