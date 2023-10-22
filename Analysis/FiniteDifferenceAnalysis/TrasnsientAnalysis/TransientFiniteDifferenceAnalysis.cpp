
//
// Created by hal9000 on 10/6/23.
//

#include "TransientFiniteDifferenceAnalysis.h"

namespace NumericalAnalysis {
    TransientFiniteDifferenceAnalysis::TransientFiniteDifferenceAnalysis(double initialTime, double stepSize, unsigned totalSteps,
            shared_ptr<TransientMathematicalProblem> mathematicalProblem,
            shared_ptr<Mesh> mesh,
            shared_ptr<Solver> solver,
            shared_ptr<NumericalIntegrator> integrationMethod,
            shared_ptr<FDSchemeSpecs> schemeSpecs,
            CoordinateType coordinateSystem) :
            FiniteDifferenceAnalysis(mathematicalProblem, std::move(mesh), std::move(solver), std::move(schemeSpecs), coordinateSystem) {

        _initialTime = initialTime;
        _finalTime = initialTime + totalSteps * stepSize;
        _stepSize = stepSize;
        _totalSteps = totalSteps;
        _transientMathematicalProblem = std::move(mathematicalProblem);
        _integrationMethod = std::move(integrationMethod);
        _integrationMethod->setSolver(this->solver);
    }
    
    void TransientFiniteDifferenceAnalysis::solve(){
        _assembleConstitutiveMatrices();
        _integrationMethod->setMatricesAndVector(_M, _C, _K, _RHS);
        _integrationMethod->setTimeParameters(_initialTime, _finalTime, _totalSteps);
        for (unsigned int i = 0; i < _totalSteps; ++i) {
            _integrationMethod->solveCurrentTimeStep(i, _initialTime + i * _stepSize, _stepSize);
            cout << "Time step " << i << " solved" << endl;
        }
        applySolutionToDegreesOfFreedom();
    }

    void TransientFiniteDifferenceAnalysis::_assembleConstitutiveMatrices() {
        auto linearSystemInitializer = TransientLinearSystemBuilder(degreesOfFreedom, mesh, _transientMathematicalProblem,
                                                                    schemeSpecs, coordinateSystem);
        linearSystemInitializer.assembleMatrices();
        _M = std::move(linearSystemInitializer.M);
        _C = std::move(linearSystemInitializer.C);
        _K = std::move(linearSystemInitializer.K);
        _RHS = std::move(linearSystemInitializer.RHS);
    }

    void TransientFiniteDifferenceAnalysis::applySolutionToDegreesOfFreedom() const {
        auto analysisResults  = _integrationMethod->results;
        
        for (auto &freeDof : *degreesOfFreedom->freeDegreesOfFreedom) {
            auto solution = analysisResults->getSolutionAtDof(0, freeDof->ID());
            freeDof->setValue(std::move(solution), 0);
            auto solutionDerivative1 = analysisResults->getSolutionAtDof(1, freeDof->ID());
            freeDof->setValue(std::move(solutionDerivative1), 1);
            auto solutionDerivative2 = analysisResults->getSolutionAtDof(2, freeDof->ID());
            freeDof->setValue(std::move(solutionDerivative2), 2);
        }
    }

    NumericalVector<double>
    TransientFiniteDifferenceAnalysis::getSolutionAtNode(NumericalVector<double> &nodeCoordinates, double tolerance,
                                                         DOFType dofType) const {
        switch (nodeCoordinates.size()) {
            case 1:
                nodeCoordinates[1] = 0.0;
                nodeCoordinates[2] = 0.0;
                break;
            case 2:
                nodeCoordinates[2] = 0.0;
                break;
            case 3:
                break;
            default:
                throw runtime_error("Target Node coordinates must have size of 1, 2, or 3");
        }
        auto coordMap = mesh->getCoordinatesToNodesMap();
        for (auto &coord : *coordMap) {
            auto iNodeCoords = coord.first;
            if (abs(iNodeCoords[0] - nodeCoordinates[0]) < tolerance &&
                abs(iNodeCoords[1] - nodeCoordinates[1]) < tolerance &&
                abs(iNodeCoords[2] - nodeCoordinates[2]) < tolerance) {
                auto node = coord.second;
                auto nodeDOFs = *node->degreesOfFreedom;
                if (dofType != NoDOFType){
                    for (auto &dof : nodeDOFs) {
                        if (dof->type() == dofType){
                            auto solution = NumericalVector<double>(_totalSteps);
                            for (auto i = 0; i < _totalSteps; ++i) {
                                solution[i] = dof->value(0, i);
                            }
                            return solution;
                        }
                    }
                }
                else{
                    if (nodeDOFs.size() != 1){
                        cout << "Warning: Node has more than one DOF, returning first element of the list" << endl;
                    }
                    auto solution = NumericalVector<double>(_totalSteps);
                    for (auto i = 0; i < _totalSteps; ++i) {
                        solution[i] = node->degreesOfFreedom->front()->value(0, i);
                    }
                    return solution;
                }
            }
        }
        throw runtime_error("Target Node not found");    }
        

}
