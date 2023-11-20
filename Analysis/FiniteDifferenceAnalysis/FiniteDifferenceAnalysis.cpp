//
// Created by hal9000 on 1/31/23.
//

#include "FiniteDifferenceAnalysis.h"

namespace NumericalAnalysis {
    
    FiniteDifferenceAnalysis::FiniteDifferenceAnalysis(shared_ptr<MathematicalProblem> mathematicalProblem,
                                                       shared_ptr<Mesh> mesh, shared_ptr<Solver> solver, shared_ptr<FiniteDifferenceSchemeOrder> schemeSpecs) :
            NumericalAnalysis(std::move(mathematicalProblem), std::move(mesh), std::move(solver)) {
        if (schemeSpecs == nullptr) {
            this->schemeSpecs = make_shared<FiniteDifferenceSchemeOrder>(2,2, this->mesh->directions());
            cout<<"yo"<<endl;
        } else {
            this->schemeSpecs = std::move(schemeSpecs);
        }
    }
    

} // NumericalAnalysis