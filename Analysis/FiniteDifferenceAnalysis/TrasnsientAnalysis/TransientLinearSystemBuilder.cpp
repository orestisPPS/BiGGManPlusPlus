//
// Created by hal9000 on 10/18/23.
//

#include "TransientLinearSystemBuilder.h"

namespace NumericalAnalysis {
    TransientLinearSystemBuilder::TransientLinearSystemBuilder(const shared_ptr<AnalysisDegreesOfFreedom>& analysisDegreesOfFreedom, const shared_ptr<Mesh> &mesh,
                                                               const shared_ptr<TransientMathematicalProblem>& mathematicalProblem, const shared_ptr<FDSchemeSpecs>& specs,
                                                               CoordinateType coordinateSystem):
                                                               EquilibriumLinearSystemBuilder(analysisDegreesOfFreedom, mesh, mathematicalProblem, specs, coordinateSystem),
                                                               _transientMathematicalProblem(mathematicalProblem) {
        
    }

    void TransientLinearSystemBuilder::assembleMatrices() {
        assembleSteadyStateLinearSystem();
        unsigned size = _analysisDegreesOfFreedom->freeDegreesOfFreedom->size();
        M = make_shared<NumericalMatrix<double>>(size, size);
        C = make_shared<NumericalMatrix<double>>(size, size);
        for (auto &dof : *_analysisDegreesOfFreedom->freeDegreesOfFreedom){
            auto dofIndex = _analysisDegreesOfFreedom->totalDegreesOfFreedomMapInverse->at(dof);
            auto dofDirection = directionOfDof.at(dof->type());
            auto dofParentNode = dof->parentNode();
            auto c = _transientMathematicalProblem->pde->temporalDerivativesCoefficients()->getTemporalCoefficient(1, dofParentNode, dofDirection);
            auto m = _transientMathematicalProblem->pde->temporalDerivativesCoefficients()->getTemporalCoefficient(2, dofParentNode, dofDirection);
            C->setElement(dofIndex, dofIndex, c);
            M->setElement(dofIndex, dofIndex, m);
        }
/*        C->printFullMatrix("C");
        M->printFullMatrix("M");
        K->printFullMatrix("K");*/
        //effectiveRHS = make_shared<NumericalVector<double>>(RHS);
    }
    
    void TransientLinearSystemBuilder::applyInitialConditions() {
        auto ic = _transientMathematicalProblem->initialConditions;
    }

} // NumericalAnalysis