//
// Created by hal9000 on 1/31/23.
//

#ifndef UNTITLED_FINITEDIFFERENCEANALYSIS_H
#define UNTITLED_FINITEDIFFERENCEANALYSIS_H

#include "../../LinearAlgebra/FiniteDifferences/FiniteDifferenceSchemeOrder.h"
#include "../NumericalAnalysis.h"

using namespace LinearAlgebra;
using namespace MathematicalEntities;
using namespace Discretization;

namespace NumericalAnalysis {

    class FiniteDifferenceAnalysis : public NumericalAnalysis {
    public:
        FiniteDifferenceAnalysis(shared_ptr<MathematicalProblem> mathematicalProblem,
                                 shared_ptr<Mesh> mesh, shared_ptr<Solver> solver, shared_ptr<FiniteDifferenceSchemeOrder> schemeSpecs);
        
        shared_ptr<FiniteDifferenceSchemeOrder> schemeSpecs;
       
    };

} // NumericalAnalysis

#endif //UNTITLED_FINITEDIFFERENCEANALYSIS_H
