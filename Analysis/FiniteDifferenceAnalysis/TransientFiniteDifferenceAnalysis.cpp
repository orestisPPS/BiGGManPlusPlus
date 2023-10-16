//
// Created by hal9000 on 10/6/23.
//

#include "TransientFiniteDifferenceAnalysis.h"

namespace NumericalAnalysis{
    TransientFiniteDifferenceAnalysis::TransientFiniteDifferenceAnalysis(
            const shared_ptr<TransientMathematicalProblem>& mathematicalProblem, const shared_ptr<Mesh>& mesh, const shared_ptr<Solver>& solver,
            const shared_ptr<FDSchemeSpecs>&schemeSpecs, CoordinateType coordinateSystem) :
            FiniteDifferenceAnalysis(mathematicalProblem, mesh, solver, schemeSpecs, coordinateSystem) {
        
        unsigned totalFreeDOF = degreesOfFreedom->freeDegreesOfFreedom->size();
        _M = make_shared<NumericalMatrix<double>>(totalFreeDOF, totalFreeDOF);
        _C = make_shared<NumericalMatrix<double>>(totalFreeDOF, totalFreeDOF);
        _K = make_shared<NumericalMatrix<double>>(totalFreeDOF, totalFreeDOF);
        
        //_linearSystemInitializer = make_unique<AnalysisLinearSystemInitializer>(
                ///degreesOfFreedom, mesh, mathematicalProblem, schemeSpecs, coordinateSystem);
        _assembleM();
        _assembleC();
        _assembleK();
    }
    
    void TransientFiniteDifferenceAnalysis::_assembleK() {
        _K = std::move(_linearSystemInitializer->linearSystem->matrix);
    }
    
    void TransientFiniteDifferenceAnalysis::_assembleC() {
        auto& properties = mathematicalProblem->pde->properties;
        switch (properties->TimePropertiesDistributionType()) {
            case Isotropic:
                double coefficient = (*properties->getLocalTimeProperties().firstOrderCoefficients)[0];
        }
        _C = std::move(_linearSystemInitializer->linearSystem->matrix);
    }
    
    void TransientFiniteDifferenceAnalysis::_assembleM() {
/*        auto properties = mathematicalProblem->pde->properties;
        switch (properties->TimePropertiesDistributionType()) {
            case Isotropic:
                double coefficient = (*properties->getLocalTimeProperties().firstOrderCoefficients)[0];
                for (auto& node : *mesh->nodes) {
                    auto& nodeDOFs = *node->degreesOfFreedom;
                    for (auto& dof : nodeDOFs) {
                        auto i = dof->index;
                        _M->at(i, i) += coefficient * dof->weight;
                    }
                }
                break;
            case Anisotropic:
                throw runtime_error("Anisotropic Time Properties not yet implemented");
        }*/
    }
}